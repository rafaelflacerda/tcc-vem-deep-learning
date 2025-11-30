import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time
from utils_data import Beam1DDataset, save_scalers
from models import BeamNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from monitor import ResourceMonitor

# --- Configurações de Caminho Robustas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Caminho para o dataset
DATASET_NAME = "dataset_viga1D_EngastadaLivre_1Meter_44Elements_100kSamples"
DATASET_PATH = os.path.join(PROJECT_ROOT, "build", "output", "ML_datasets", DATASET_NAME)
TRAIN_FILE = os.path.join(DATASET_PATH, "training_data.csv")

# --- Configurações ---
EPOCHS = 1000
BATCH_SIZE = 16384  # 🔥 AUMENTADO para aproveitar GPU
LR = 1e-3
LAMBDA_THETA = 10.0

# --- Configuração de Hardware ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎 Apple Silicon (MPS)")
else:
    DEVICE = "cpu"
    print("⚠️  CPU")

# ============================================================
# ✅ FUNÇÃO DE PERDA: SOBOLEV DE 1ª ORDEM
# ============================================================
def sobolev_loss(y_pred, y_true, lambda_theta=1.0):
    """
    Calcula o erro do valor (w) e o erro da derivada (theta) separadamente.
    """
    w_pred, theta_pred = y_pred[:, 0], y_pred[:, 1]
    w_true, theta_true = y_true[:, 0], y_true[:, 1]
    
    loss_w = torch.mean((w_pred - w_true) ** 2)
    loss_theta = torch.mean((theta_pred - theta_true) ** 2)
    
    return loss_w + (lambda_theta * loss_theta)

def main():
    print(f"\n{'='*70}")
    print(f"🚀 TREINAMENTO | Batch: {BATCH_SIZE} | LR: {LR} | Lambda: {LAMBDA_THETA}")
    print(f"{'='*70}\n")
    
    # 1. Carregar Dados
    full_dataset = Beam1DDataset(TRAIN_FILE, is_training=True)
    save_scalers(full_dataset.scaler_X, full_dataset.scaler_y, f"{DATASET_PATH}/scalers.pkl")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4
    )
    
    print(f"📊 Dataset: {len(full_dataset)} pontos | Treino: {train_size} | Val: {val_size}")
    print(f"📦 Batches/época: {len(train_loader)}\n")

    # 2. Modelo e Otimizador
    model = BeamNet(input_dim=6, output_dim=2, hidden_dim=1024).to(DEVICE)
    print(f"✅ Modelo: hidden_dim=1024 | Device: {next(model.parameters()).device}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-6)
    scaler = GradScaler('cuda')
    
    # 3. Inicializar Monitor
    monitor = ResourceMonitor(log_file=f"{DATASET_PATH}/training_stats.json")
    
    loss_history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    total_batches = len(train_loader)

    # 4. Loop de Treino
    print(f"🏋️  Iniciando {EPOCHS} épocas...\n")
    print(f"{'='*70}")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        batch_start_time = time.time()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            data_load_time = time.time() - batch_start_time
            compute_start_time = time.time()
            
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                y_pred = model(X_batch)
                loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            compute_time = time.time() - compute_start_time
            
            monitor.log_batch_detailed(data_load_time, compute_time)
            
            # 📊 Log COMPACTO a cada 50 batches
            if batch_idx % 50 == 0 and batch_idx > 0:
                monitor.print_compact(epoch+1, batch_idx, total_batches, BATCH_SIZE)
            
            batch_start_time = time.time()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                
                with autocast('cuda'):
                    y_pred = model(X_batch)
                    loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        
        # 💾 Salvar melhor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{DATASET_PATH}/beamnet_model_best.pth")
            print(f"💾 Melhor modelo | Val Loss: {avg_val_loss:.6f}")
        
        # Log de época a cada 10 épocas
        if (epoch+1) % 10 == 0:
            print(f"\n📈 Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.2e}\n")

    # 5. Finalização
    print(f"\n{'='*70}")
    print(f"🏁 TREINAMENTO CONCLUÍDO")
    print(f"{'='*70}\n")
    
    monitor.diagnose_bottleneck()
    monitor.save_excel_report(output_path=f"{DATASET_PATH}/training_report.csv")

    # 6. Salvar Modelo Final
    torch.save(model.state_dict(), f"{DATASET_PATH}/beamnet_model.pth")
    print(f"✅ Modelo final: {DATASET_PATH}/beamnet_model.pth")

    # 7. Plotar Loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history['train'], label='Treino', linewidth=2)
    plt.plot(loss_history['val'], label='Validação', linewidth=2)
    plt.yscale('log')
    plt.title(f'Curva de Convergência (Sobolev Lambda={LAMBDA_THETA})', fontsize=14)
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss Ponderada (log scale)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{DATASET_PATH}/training_loss.png", dpi=300)
    print(f"✅ Gráfico: {DATASET_PATH}/training_loss.png\n")

if __name__ == "__main__":
    main()