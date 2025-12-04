import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from models import BeamNet, BeamNetLarge
from dataloader import OptimizedBeamDataset, save_scalers
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, SequentialLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

# --- Configurações de Caminho Robustas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Caminho para o dataset
N_SAMPLES = 2500
NPZ_FILE = os.path.join(PROJECT_ROOT, "00_PROBLEMA_UNIDIMENSIONAL", "dataset", "npz", f"beam_dataset_{N_SAMPLES}_samples.npz")

DATASET_NAME = "dataset_viga1D_" + str(N_SAMPLES) + "_samples"
DATASET_PATH = os.path.join(PROJECT_ROOT, "00_PROBLEMA_UNIDIMENSIONAL", "treinamentos", DATASET_NAME)

os.makedirs(DATASET_PATH, exist_ok=True)

# --- Configurações ---
EPOCHS = 1000
BATCH_SIZE = 256 #256

LR = 1e-3
FACTOR = 0.5
PATIENCE = 50 # testar 25
MIN_LR = 1e-6

LAMBDA_THETA = 10.0 # Peso da Derivada (Sobolev): Aumente se a rotação estiver imprecisa

HIDDEN_DIM = 512
DROPOUT_P = 0.15 # TESTAR 0.08 e 0.12

# --- Configuração de Hardware ---
DEVICE = "cuda"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

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

def main():
    print(f" Iniciando treinamento em: {DEVICE}")
    print(f" Usando Sobolev Loss com Lambda_Theta = {LAMBDA_THETA}")
    
    # 1. Carregar Dados
    full_dataset = OptimizedBeamDataset(NPZ_FILE)
    save_scalers(full_dataset.X_mean, full_dataset.X_std, 
             full_dataset.y_mean, full_dataset.y_std,
             f"{DATASET_PATH}/scalers.pkl")
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f" Dados: {len(full_dataset)} pontos totais")
    print(f" Treino: {train_size} | Validação Interna: {val_size}")

    # 2. Modelo e Otimizador
    model = BeamNet(input_dim=5, output_dim=2, hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P).to(DEVICE)
    #odel = BeamNetLarge(input_dim=5, output_dim=2, hidden_dim=HIDDEN_DIM, dropout_p=DROPOUT_P).to(DEVICE)
    model = torch.compile(model, mode="max-autotune")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    scaler = GradScaler()
    
    # ============================================================
    # SCHEDULERS: WARMUP + MAIN
    # ============================================================
    warmup_epochs = 1  # ✅ CORRIGIDO: 10 épocas é suficiente
    
    # Warmup: vai de 0 até 1.0 (multiplicando o LR base)
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)
    )
    
    # Main scheduler: ReduceLROnPlateau para ajuste fino
    main_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=FACTOR, 
        patience=PATIENCE, 
        min_lr=MIN_LR
    )
    
    loss_history = {'train': [], 'val': []}

    # 3. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
    
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.cuda(non_blocking=True)
            y_batch = y_batch.cuda(non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(dtype=torch.bfloat16):
                y_pred = model(X_batch)
                loss = sobolev_loss(y_pred, y_batch, lambda_theta = LAMBDA_THETA)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validação
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.cuda(non_blocking=True)
                y_batch = y_batch.cuda(non_blocking=True)
                
                with autocast(dtype=torch.bfloat16):
                    y_pred = model(X_batch)
                    loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)
                    
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # ============================================================
        # ✅ SCHEDULER CORRIGIDO: ESCOLHER APENAS UM POR ÉPOCA
        # ============================================================
        if epoch < warmup_epochs:
            warmup_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f" Warmup Epoch {epoch+1}/{warmup_epochs} | LR: {current_lr:.2e}")
        else:
            main_scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

    # 4. Salvar Modelo
    torch.save(model.state_dict(), f"{DATASET_PATH}/beamnet_model.pth")
    print("💾 Modelo salvo!")

    # 5. Plotar Loss
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
    main()