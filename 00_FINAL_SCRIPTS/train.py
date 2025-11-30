import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import sys

# Adicionar pasta ao path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))

from model import BayesianVEMNet
from dataloader import get_dataloaders

# ==================== CONFIGURAÇÃO ====================
# Dataset
N_SAMPLES = int(input("Quantos samples? (5, 10, 100, 1000, 10000): "))
NPZ_FILE = PROJECT_ROOT / f"00_URGENTE/malha/training_dataset_npz/meshes_{N_SAMPLES}_samples.npz"
DATASET_NAME = f"meshes_{N_SAMPLES}_samples"

print(f"\n✓ Dataset selecionado: {DATASET_NAME}\n")

# Hiperparâmetros
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 200
NUM_WORKERS = 8

# Early Stopping (deixe False para desativar)
EARLY_STOPPING = True
PATIENCE = 20  # Parar se não melhorar por N epochs

# Checkpointing
SAVE_CHECKPOINT_EVERY = 50  # Salvar checkpoint a cada N epochs

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== SETUP ====================
# Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{DATASET_NAME}_{timestamp}"

# Criar pasta do experimento
experiment_dir = PROJECT_ROOT / "models" / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

# Arquivo de log
log_file = experiment_dir / "training_log.txt"

def log_print(message, file=log_file):
    """Print e salva no log"""
    print(message)
    with open(file, 'a') as f:
        f.write(message + '\n')

# ==================== CABEÇALHO DO LOG ====================
header = f"""
{'='*60}
VEM Deep Learning Training
{'='*60}
Data/Hora Início: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {DATASET_NAME}.npz
Experimento: {experiment_name}

HIPERPARÂMETROS:
  Arquitetura: 6 camadas [256-256-128-128-64-64]
  Batch size: {BATCH_SIZE}
  Learning rate: {LEARNING_RATE}
  Optimizer: AdamW (weight_decay={WEIGHT_DECAY})
  Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
  Epochs: {EPOCHS}
  Precision: FP16
  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
  Early Stopping: {'Ativo (patience=' + str(PATIENCE) + ')' if EARLY_STOPPING else 'Desativado'}
  
{'='*60}
"""

log_print(header)

# ==================== CARREGAR DADOS ====================
log_print("\nCarregando dados...")
train_loader, val_loader = get_dataloaders(
    npz_path=str(NPZ_FILE),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_split=0.8,
    seed=42
)

# ==================== CRIAR MODELO ====================
log_print("\nCriando modelo...")
model = BayesianVEMNet(dropout_rate=0.2).to(DEVICE)
log_print(f"Parâmetros treináveis: {model.count_parameters():,}")

# torch.compile (otimização L4)
log_print("Compilando modelo (torch.compile)...")
model = torch.compile(model)

# ==================== OPTIMIZER & SCHEDULER ====================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

# Mixed Precision (FP16)
scaler = GradScaler()

# Loss function
criterion = nn.MSELoss()

# ==================== TRACKING ====================
best_val_loss = float('inf')
epochs_without_improvement = 0
train_losses = []
val_losses = []

# Cabeçalho da tabela
log_print("\n" + "="*80)
log_print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'LR':<12}{'Time(s)':<10}{'Best':<5}")
log_print("="*80)

# ==================== LOOP DE TREINO ====================
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # ========== TRAIN ==========
    model.train()
    train_loss = 0.0
    
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward com FP16
        with autocast():
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
        
        # Backward com scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
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
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            
            with autocast():
                Y_pred = model(X_batch)
                loss = criterion(Y_pred, Y_batch)
            
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Scheduler step
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # ========== LOGGING ==========
    epoch_time = time.time() - epoch_start
    is_best = val_loss < best_val_loss
    best_marker = '*' if is_best else ''
    
    log_print(
        f"{epoch:<8}{train_loss:<15.6f}{val_loss:<15.6f}"
        f"{current_lr:<12.6f}{epoch_time:<10.1f}{best_marker:<5}"
    )
    
    # ========== CHECKPOINTING ==========
    # Salvar best model
    if is_best:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(
            model.state_dict(),
            experiment_dir / "best_model.pth"
        )
    else:
        epochs_without_improvement += 1
    
    # Salvar last model (sempre sobrescreve)
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
    
    # Salvar checkpoint periódico
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
    
    # ========== EARLY STOPPING ==========
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