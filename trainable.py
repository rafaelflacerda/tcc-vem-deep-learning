import torch
from torch.utils.data import DataLoader
import os

from models import BeamNet
from dataloader import PreprocessedBeamDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from ray.air import session

# --- Configurações de Caminho Robustas ---
# Caminho raiz sempre consistente
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Pastas fixas
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

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


def trainable(config):
    
    n_samples = config["n_samples"]
    
    PREPROC_NPZ = os.path.join(
        DATASET_DIR,
        f"beam_dataset_{n_samples}_samples_preproc.npz"
    )

    SCALERS_NPZ = os.path.join(
        DATASET_DIR,
        f"beam_scalers_{n_samples}_samples.npz"
    )
    
    torch.backends.cudnn.benchmark = True
    
    # Pega LR e batch_size da linha de comando
    LR = config["LR_inicial"]
    BATCH_SIZE = config["batch_size"]
    LAMBDA_THETA = config["lambda_theta"]
    HIDDEN_DIM = config["hidden_dim"]
    DROPOUT_P = 0.0
    WEIGHT_DECAY = config["weight_decay"]
    EPOCHS = config["epochs"]
    
    # --------------------------------------------------
    # 1. Carregar Dados
    # --------------------------------------------------
    full_dataset = PreprocessedBeamDataset(
        PREPROC_NPZ,
        scalers_path=SCALERS_NPZ  # opcional, mas útil se você quiser usar depois
    )

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # --------------------------------------------------
    # 2. DataLoaders
    # --------------------------------------------------
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
        eta_min=LR/50    # LR mínimo
    )

    def sched_step():
        scheduler.step()

    # 6. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # -----------------------------
            # Data + to(device)
            # -----------------------------
            X_batch = X_batch.cuda(non_blocking=True)
            y_batch = y_batch.cuda(non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # -----------------------------
            # Forward
            # -----------------------------
            with autocast("cuda", dtype=torch.bfloat16):
                y_pred = model(X_batch)
                loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)

            # -----------------------------
            # Backward
            # -----------------------------
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # -----------------------------
            # Step do otimizador
            # -----------------------------
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

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
        
        # --- AQUI entra o session.report --- #
        session.report({
            "val_loss": avg_val_loss,
            "val_w_mse": avg_val_w_mse,
            "train_loss": avg_train_loss,
            "training_iteration": epoch + 1,
            "lr": current_lr,
        })
        # --- FIM DO session.report --- #
        
    return {"val_loss": avg_val_loss, "val_w_mse": avg_val_w_mse}