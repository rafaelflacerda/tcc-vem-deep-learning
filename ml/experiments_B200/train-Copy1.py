import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
import argparse

from models import BeamNet, BeamNetLarge
from dataloader import PreprocessedBeamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

# --- Configurações de Caminho Robustas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Caminho para o dataset
N_SAMPLES = 10000

DATASET_NAME = "dataset_viga1D_" + str(N_SAMPLES) + "_samples"
DATASET_PATH = os.path.join(
    "/workspace",
    "treinamentos",
    DATASET_NAME
)

os.makedirs(DATASET_PATH, exist_ok=True)

RAW_NPZ_DIR = "/workspace/dataset/npz"
PREPROC_DIR = "/workspace/dataset/npz_preprocessed"

PREPROC_NPZ = os.path.join(
    PREPROC_DIR,
    f"beam_dataset_{N_SAMPLES}_samples_preproc.npz"
)

SCALERS_NPZ = os.path.join(
    PREPROC_DIR,
    f"beam_scalers_{N_SAMPLES}_samples.npz"
)

# --- Configuração de Hardware ---
DEVICE = "cuda"

# ============================================================
# NOVA FUNÇÃO DE PERDA: SOBOLEV DE 1ª ORDEM
# ============================================================
def sobolev_loss(y_pred, y_true, lambda_theta=1.0):
    """
    Calcula o erro do valor (w) e o erro da derivada (theta) separadamente.
    
    Args:
        y_pred: Tensor [batch, 2] -> (w_pred, theta_pred)
        y_true: Tensor [batch, 2] -> (w_true, theta_true)
        lambda_theta: Peso dado ao erro da derivada (rotação).
    """
    # Separa as colunas (0 = Deslocamento, 1 = Rotação/Derivada)
    w_pred, theta_pred = y_pred[:, 0], y_pred[:, 1]
    w_true, theta_true = y_true[:, 0], y_true[:, 1]
    
    # Calcula MSE individualmente
    loss_w = torch.mean((w_pred - w_true) ** 2)
    loss_theta = torch.mean((theta_pred - theta_true) ** 2)
    
    # Retorna a soma ponderada
    return loss_w + (lambda_theta * loss_theta)


def main(args):
    # Pega LR e batch_size da linha de comando
    LR = args.LR_inicial
    BATCH_SIZE = args.batch_size
    MIN_LR = LR / 50
    LAMBDA_THETA = args.lambda_theta
    HIDDEN_DIM = args.hidden_dim
    DROPOUT_P = args.dropout
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs


    print(f" Iniciando treinamento em: {DEVICE}")
    print(f" Usando Sobolev Loss com Lambda_Theta = {LAMBDA_THETA}")
    print(f" Configuração: N_SAMPLES={N_SAMPLES} | batch_size={BATCH_SIZE} | LR_inicial={LR:.2e}")
    
    # --------------------------------------------------
    # 1. Carregar Dados (medir tempo)
    # --------------------------------------------------
    t0 = time.perf_counter()
    full_dataset = PreprocessedBeamDataset(
        PREPROC_NPZ,
        scalers_path=SCALERS_NPZ  # opcional, mas útil se você quiser usar depois
    )
    t1 = time.perf_counter()
    print(f"[TIMER] Carregar PreprocessedBeamDataset: {t1 - t0:.3f} s")

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # --------------------------------------------------
    # 2. DataLoaders (medir tempo)
    # --------------------------------------------------
    t2 = time.perf_counter()
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
    )
    t3 = time.perf_counter()
    print(f"[TIMER] Criar DataLoaders: {t3 - t2:.3f} s")
    
    print(f" Dados: {len(full_dataset)} pontos totais")
    print(f" Treino: {train_size} | Validação Interna: {val_size}")

    # --------------------------------------------------
    # 3. Pegar primeiro batch (pra medir overhead inicial)
    # --------------------------------------------------
    t4 = time.perf_counter()
    first_batch = next(iter(train_loader))
    t5 = time.perf_counter()
    print(f"[TIMER] Pegar primeiro batch do DataLoader: {t5 - t4:.3f} s")

    # 4. Modelo e Otimizador
    model = BeamNet(
        input_dim=5,
        output_dim=2,
        hidden_dim=HIDDEN_DIM,
        dropout_p=DROPOUT_P
    ).to(DEVICE)
    # model = BeamNetLarge(...)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scaler = GradScaler("cuda")

    # Scheduler principal: Cosine Annealing
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,     # período completo do cosseno
        eta_min=MIN_LR    # LR mínimo
    )

    def sched_step():
        scheduler.step()

    # --------------------------------------------------
    # 5. Primeiro forward na GPU (init CUDA/cuDNN)
    # --------------------------------------------------
    X_test, _ = first_batch
    X_test = X_test.cuda(non_blocking=True)

    t6 = time.perf_counter()
    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            _ = model(X_test)
    torch.cuda.synchronize()
    t7 = time.perf_counter()
    print(f"[TIMER] Primeiro forward na GPU (init CUDA/cuDNN): {t7 - t6:.3f} s")

    loss_history = {'train': [], 'val': []}
    epoch_times = []

    # métricas globais que queremos no final
    best_val_sobolev_loss = float("inf")
    best_val_w_mse = float("inf")
    epoch_of_best_val = -1

    final_val_sobolev_loss = None
    final_val_w_mse = None

    # 6. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        # timers por época
        data_time = 0.0
        fwd_time = 0.0
        bwd_time = 0.0
        step_time = 0.0

        epoch_start = time.perf_counter()
        
        for X_batch, y_batch in train_loader:
            # -----------------------------
            # Data + to(device)
            # -----------------------------
            t_d0 = time.perf_counter()
            X_batch = X_batch.cuda(non_blocking=True)
            y_batch = y_batch.cuda(non_blocking=True)
            torch.cuda.synchronize()
            t_d1 = time.perf_counter()
            data_time += (t_d1 - t_d0)
            
            optimizer.zero_grad(set_to_none=True)
            
            # -----------------------------
            # Forward
            # -----------------------------
            t_f0 = time.perf_counter()
            with autocast("cuda", dtype=torch.bfloat16):
                y_pred = model(X_batch)
                loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)
            torch.cuda.synchronize()
            t_f1 = time.perf_counter()
            fwd_time += (t_f1 - t_f0)

            # -----------------------------
            # Backward
            # -----------------------------
            t_b0 = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.cuda.synchronize()
            t_b1 = time.perf_counter()
            bwd_time += (t_b1 - t_b0)

            # -----------------------------
            # Step do otimizador
            # -----------------------------
            t_s0 = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            t_s1 = time.perf_counter()
            step_time += (t_s1 - t_s0)
            
            train_loss += loss.item()
            
        epoch_end = time.perf_counter()
        total_epoch_time = epoch_end - epoch_start
        epoch_times.append(total_epoch_time)

        avg_train_loss = train_loss / len(train_loader)

        # --------------------------------------------------
        # Validação + MSE só de w
        # --------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_w_mse_sum = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.cuda(non_blocking=True)
                y_batch = y_batch.cuda(non_blocking=True)
                
                with autocast("cuda", dtype=torch.bfloat16):
                    y_pred = model(X_batch)
                    loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)

                    # MSE apenas de w (coluna 0)
                    w_pred = y_pred[:, 0].float()
                    w_true = y_batch[:, 0].float()
                    mse_w_batch = torch.mean((w_pred - w_true) ** 2)
                    
                val_loss += loss.item()
                val_w_mse_sum += mse_w_batch.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_w_mse = val_w_mse_sum / len(val_loader)

        # Atualiza scheduler
        sched_step()
        current_lr = optimizer.param_groups[0]['lr']
        
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)

        # Atualiza métricas globais (melhor época)
        if avg_val_loss < best_val_sobolev_loss:
            best_val_sobolev_loss = avg_val_loss
            best_val_w_mse = avg_val_w_mse
            epoch_of_best_val = epoch + 1  # 1-based

        # Atualiza métricas finais (última época)
        final_val_sobolev_loss = avg_val_loss
        final_val_w_mse = avg_val_w_mse

        # --------------------------------------------------
        # Print de tempos + perdas + LR por época
        # --------------------------------------------------
        print(
            f"[Epoch {epoch+1:04d}/{EPOCHS}] "
            f"train_loss: {avg_train_loss:.6e} | "
            f"val_loss: {avg_val_loss:.6e} | "
            f"val_w_mse: {avg_val_w_mse:.6e} | "
            f"LR: {current_lr:.2e} || "
            f"total: {total_epoch_time:.3f}s | "
            f"data: {data_time:.3f}s | "
            f"fwd: {fwd_time:.3f}s | "
            f"bwd: {bwd_time:.3f}s | "
            f"step: {step_time:.3f}s"
        )

    # --------------------------------------------------
    # 7. Métricas agregadas no final
    # --------------------------------------------------
    tempo_medio_por_epoch = sum(epoch_times) / len(epoch_times)

    print("\n================= RESUMO DO EXPERIMENTO =================")
    print(f"batch_size = {BATCH_SIZE}")
    print(f"LR_inicial = {LR:.4e}")
    print(f"best_val_sobolev_loss = {best_val_sobolev_loss:.6e}")
    print(f"best_val_w_mse        = {best_val_w_mse:.6e}")
    print(f"epoch_of_best_val     = {epoch_of_best_val}")
    print(f"final_val_sobolev_loss = {final_val_sobolev_loss:.6e}")
    print(f"final_val_w_mse        = {final_val_w_mse:.6e}")
    print(f"tempo_medio_por_epoch  = {tempo_medio_por_epoch:.3f} s")
    print("=========================================================\n")

    # >>> ÚNICA MUDANÇA PEDIDA: LINHA NO FORMATO CSV <<<
    print("best_val_sobolev_loss;best_val_w_mse;epoch_of_best_val;final_val_sobolev_loss;final_val_w_mse;tempo_medio_por_epoch")
    print(f"{best_val_sobolev_loss};{best_val_w_mse};{epoch_of_best_val};{final_val_sobolev_loss};{final_val_w_mse};{tempo_medio_por_epoch}")

    # 8. Salvar Modelo
    torch.save(model.state_dict(), f"{DATASET_PATH}/beamnet_model.pth")
    print("💾 Modelo salvo!")

    # 9. Plotar Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='Treino')
    plt.plot(loss_history['val'], label='Validação')
    plt.yscale('log')
    plt.title(f'Curva de Convergência (Sobolev Lambda = {LAMBDA_THETA})')
    plt.xlabel('Épocas')
    plt.ylabel('Loss Ponderada')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"{DATASET_PATH}/training_loss.png")
    print("📊 Gráfico de Loss salvo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--LR_inicial",
        type=float,
        default=0.0014,
        help="Learning rate inicial (float), ex: 3e-4"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Tamanho do batch (int), ex: 256"
    )
    parser.add_argument(
        "--lambda_theta",
        type=float,
        default=5
    )
    parser.add_argument(
        "--hidden_dim",
        type = int,
        default = 384
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--weight_decay",
        type = float,
        default = 1e-4
    )
    parser.add_argument(
        "--epochs",
        type = int,
        default = 200
    )
     
    args = parser.parse_args()
    main(args)
