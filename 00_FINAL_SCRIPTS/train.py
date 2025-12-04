import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import sys
import argparse

# Adicionar pasta ao path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from model import BayesianVEMNet
from dataloader import get_dataloaders
from logging_utils import setup_experiment_logging
from sobolev_utils import compute_element_centroids, build_sigma_inputs_for_case, build_C_plane_stress, compute_sigma_pred_for_case, sobolev_stress_loss_for_case
# (deixa comentado por enquanto, só vamos usar quando plugar a loss de Sobolev)

# ==================== CONFIGURAÇÃO ====================
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--n_samples", type=int, default=1000)
args = parser.parse_args()

N_SAMPLES = args.n_samples
NPZ_FILE = PROJECT_ROOT / f"00_URGENTE/malha/training_dataset_npz/meshes_{N_SAMPLES}_samples.npz"
DATASET_NAME = f"meshes_{N_SAMPLES}_samples"

print(f"\n✓ Dataset selecionado: {DATASET_NAME}\n")

BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
WEIGHT_DECAY = 1e-5
EPOCHS = 400
NUM_WORKERS = 7

EARLY_STOPPING = False
PATIENCE = 20

SAVE_CHECKPOINT_EVERY = 50

# Peso da loss de tensões (Sobolev)
LAMBDA_SIGMA = 0.1  # pode começar pequeno e ir ajustando

# Número máximo de casos por epoch para aplicar Sobolev (para controle, se quiser)
MAX_SOB_CASES_PER_EPOCH = None  # ou um int, tipo 10 se quiser limitar

if not torch.cuda.is_available():
    raise RuntimeError("❌ Nenhuma GPU CUDA disponível! Abortando execução.")
DEVICE = torch.device("cuda")

# ==================== LOGGING / EXPERIMENTO ====================
experiment_dir, experiment_name, log_print = setup_experiment_logging(
    project_root=PROJECT_ROOT,
    dataset_name=DATASET_NAME,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    epochs=EPOCHS,
    early_stopping=EARLY_STOPPING,
    patience=PATIENCE
)

# ==================== CARREGAR DADOS ====================
log_print("\nCarregando dados...")
train_loader, val_loader = get_dataloaders(
    npz_path=str(NPZ_FILE),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_split=0.8,
    seed=42
)

has_sobolev = (
    hasattr(train_loader, "case_nodes")
    and train_loader.case_nodes is not None
    and hasattr(train_loader, "case_sigma")
    and train_loader.case_sigma is not None
)

if has_sobolev:
    log_print("Sobolev: tensões + malha disponíveis. Loss de Sobolev será usada no treino.")
else:
    log_print("Sobolev: tensões/malha NÃO disponíveis. Treino apenas com MSE de deslocamento.")

# ==================== CRIAR MODELO ====================
log_print("\nCriando modelo...")
model = BayesianVEMNet(dropout_rate=0.2).to(DEVICE)
log_print(f"Parâmetros treináveis: {model.count_parameters():,}")

log_print("Compilando modelo (torch.compile)...")
model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

scaler = GradScaler()
criterion = nn.MSELoss()

best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses = []
val_losses = []

# ==================== LOOP DE TREINO ====================
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # ========== TRAIN ==========
    model.train()
    train_loss = 0.0
    
    for X_batch, Y_batch, case_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        X_batch.requires_grad_(True)
        
        Y_batch = Y_batch.to(DEVICE)
        
        case_batch = case_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            Y_pred = model(X_batch)
            loss_u = criterion(Y_pred, Y_batch)  # (por enquanto só MSE de deslocamento)
            
        loss = loss_u
        
                # -------- Loss de tensões (Sobolev) por caso --------
        if LAMBDA_SIGMA > 0.0 and train_loader.case_sigma is not None:
            unique_cases_batch = case_batch.unique()
            loss_sigma_total = 0.0
            n_cases_sigma = 0

            for c in unique_cases_batch:
                c_int = int(c.item())
                if c_int not in train_loader.case_sigma:
                    continue

                # Desliga o autocast aqui para não ter doideira com autograd de 2ª ordem
                with torch.cuda.amp.autocast(enabled=False):
                    loss_sigma_c = sobolev_stress_loss_for_case(
                        model=model,
                        case_id=c_int,
                        loader=train_loader,
                        device=DEVICE,
                        reduction="mean",
                    )
                loss_sigma_total = loss_sigma_total + loss_sigma_c
                n_cases_sigma += 1

            if n_cases_sigma > 0:
                loss_sigma_mean = loss_sigma_total / n_cases_sigma
                loss = loss + LAMBDA_SIGMA * loss_sigma_mean
        # ----------------------------------------------------
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, Y_batch, case_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            
            with autocast():
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    epoch_time = time.time() - epoch_start
    is_best = val_loss < best_val_loss
    best_marker = '*' if is_best else ''
    
    log_print(
        f"{epoch:<8}{train_loss:<15.6f}{val_loss:<15.6f}"
        f"{current_lr:<12.6f}{epoch_time:<10.1f}{best_marker:<5}"
    )
    
    # ========== CHECKPOINTING ==========
    if is_best:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(
            model.state_dict(),
            experiment_dir / "best_model.pth"
        )
    else:
        epochs_without_improvement += 1
    
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, experiment_dir / "last_model.pth")
    
    if epoch % SAVE_CHECKPOINT_EVERY == 0:
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, experiment_dir / f"checkpoint_epoch_{epoch}.pth")
    
    if EARLY_STOPPING and epochs_without_improvement >= PATIENCE:
        log_print("\n" + "="*80)
        log_print(f"Early stopping! Sem melhora por {PATIENCE} epochs.")
        log_print(f"Melhor val_loss: {best_val_loss:.6f} (epoch {epoch - PATIENCE})")
        break

# ==================== FINALIZAÇÃO ====================
total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

footer = f"""
{'='*80}
Treino Finalizado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Duração Total: {hours}h {minutes}min {seconds}s
Melhor val_loss: {best_val_loss:.6f}
Modelo salvo em: {experiment_dir / 'best_model.pth'}
{'='*80}
"""

log_print(footer)
log_print(f"\n✓ Experimento completo: {experiment_name}")