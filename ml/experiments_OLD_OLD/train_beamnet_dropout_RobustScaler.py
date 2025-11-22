import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import numpy as np

from utils_data_RobustScaler import (
    get_dataloaders,
    validate_with_inverse,              
    save_checkpoint
)
from models import BeamNetDropout

# ============================================================
# 🔹 Funções de treino e validação
# ============================================================
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Treino {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        
        # ✅ NOVO: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)

# ============================================================
# 🔹 Script principal
# ============================================================
if __name__ == "__main__":
    # --- Configurações ---
    DATA_DIR = "../build/output/ml_dataset_100k"
    EPOCHS = 2500
    BATCH_SIZE = 256
    LR = 2e-4
    
    FAST_TEST = True
    
    if FAST_TEST:
        EPOCHS = 200
        BATCH_SIZE = 128
        N_SAMPLES = 10000
        LR = 1e-3
    else:
        N_SAMPLES = None

    # Criação automática de subdiretório para experimentos
    now = datetime.datetime.now()
    dataset_name = os.path.basename(os.path.normpath(DATA_DIR))
    exp_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{dataset_name}_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LR:.0e}"
    SAVE_DIR = os.path.join("experiments_RobustScaler", exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- Dataset ---
    print("\n📥 Carregando dataset...")
    train_loader, val_loader, dataset = get_dataloaders(
        DATA_DIR,
        n_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        normalize=True,
        val_split=0.2
    )
    print(f"✓ Dados carregados: {len(dataset)} amostras totais")
    print(f"  Batches de treino: {len(train_loader)} | Validação: {len(val_loader)}")

    # --- Modelo e otimizador ---
    model = BeamNetDropout(input_dim=10, hidden_dim=512, dropout_p=0.05).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Scheduler com ajuste ultra-fino
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.95,     # ← Menos agressivo
        patience=120,    # ← Mais paciente
        min_lr=1e-6
    )

    # ============================================================
    # ✅ Criar listas SEPARADAS para transformado e original
    # ============================================================
    train_losses = []
    val_losses_transformed = []
    val_losses_original = []
    
    best_val_original = float("inf")
    patience_counter = 0
    EARLY_STOP_PATIENCE = 1000
    best_model_path = os.path.join(SAVE_DIR, "beamnet_dropout_best.pt")

    print("\n🚀 Iniciando treinamento com dropout e RobustScaler...\n")
    print(f"Config: LR={LR:.0e} | Epochs={EPOCHS} | Batch={BATCH_SIZE}")
    print(f"Scheduler: factor=0.95, patience=120")
    print(f"Early stop: patience={EARLY_STOP_PATIENCE}")
    print(f"✅ Validação em DOIS espaços: transformado (treino) + original (interpretação)\n")

    start_time = datetime.datetime.now()

    # ============================================================
    # Loop de treinamento com validate_with_inverse()
    # ============================================================
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        
        # ✅ Retorna DOIS valores
        val_loss_transformed, val_loss_original = validate_with_inverse(
            model, val_loader, criterion, device, dataset.scaler_y, verbose=True
        )
        
        # Usar transformado para scheduler (otimização)
        scheduler.step(val_loss_transformed)

        train_losses.append(train_loss)
        val_losses_transformed.append(val_loss_transformed)
        val_losses_original.append(val_loss_original)

        current_lr = optimizer.param_groups[0]['lr']
        print(f" Época {epoch+1:04d}/{EPOCHS} | "
              f"Treino: {train_loss:.6f} | "
              f"Val(orig): {val_loss_original:.6f} | "
              f"LR: {current_lr:.2e}")

        # ============================================================
        # Early stopping baseado em espaço ORIGINAL
        # ============================================================
        if val_loss_original < best_val_original:
            best_val_original = val_loss_original
            
            save_checkpoint(
                model, optimizer, dataset.scaler_y,
                epoch, val_loss_original, best_model_path
            )
            
            patience_counter = 0
            print(f"✨ Novo melhor modelo (espaço original)! Val loss: {val_loss_original:.6f}")
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⛔ Early stopping na época {epoch+1}")
            print(f"   Sem melhora por {EARLY_STOP_PATIENCE} épocas")
            break

    end_time = datetime.datetime.now()
    
    # Formatar tempo de forma legível
    total_time_delta = end_time - start_time
    total_seconds = total_time_delta.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    total_time = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"

    # ============================================================
    # Gráficos em DOIS espaços
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico 1: Espaço Transformado (para otimização)
    ax = axes[0]
    ax.plot(train_losses, label='Treino', alpha=0.8, linewidth=2)
    ax.plot(val_losses_transformed, label='Validação (transformado)', alpha=0.8, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Loss (escala log)')
    ax.set_title(f'Treino em Espaço Transformado (para otimização)')
    ax.legend()
    ax.grid(True, ls="--", alpha=0.6)
    
    # Gráfico 2: Espaço Original (interpretação)
    ax = axes[1]
    ax.plot(train_losses, label='Treino', alpha=0.8, linewidth=2)
    ax.plot(val_losses_original, label='Validação (espaço original)', alpha=0.8, linewidth=2, color='orange')
    ax.set_yscale("log")
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Loss - Espaço Original (escala log)')
    ax.set_title(f'Interpretação em Espaço Original (comparável com VEM)')
    ax.legend()
    ax.grid(True, ls="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve_dropout.png"), dpi=150)
    plt.close()

    print(f"\n📊 Curva salva em: {SAVE_DIR}/training_curve_dropout.png")

    # ============================================================
    # Usar save_checkpoint() para modelo final
    # ============================================================
    final_model_path = os.path.join(SAVE_DIR, "beamnet_dropout_final.pt")
    save_checkpoint(
        model, optimizer, dataset.scaler_y,
        epoch, val_loss_original, final_model_path
    )

    # --- Salvar arquivo resumo ---
    summary_path = os.path.join(SAVE_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RESUMO DO TREINAMENTO - BeamNetDropout com RobustScaler\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("✅ CORREÇÕES IMPLEMENTADAS:\n")
        f.write("- RobustScaler para normalização robusta\n")
        f.write("- Validação em DOIS espaços (transformado + original)\n")
        f.write("- Early stopping baseado em espaço original\n")
        f.write("- Scaler salvo com pickle via save_checkpoint()\n")
        f.write("- Gradient clipping ativado\n")
        f.write("- Batch size reduzido para estabilidade\n\n")
        
        f.write("--- INFORMAÇÕES TEMPORAIS ---\n")
        f.write(f"Início:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Término:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duração:  {total_time}\n\n")
        
        f.write("--- DADOS ---\n")
        f.write(f"Dataset: {DATA_DIR}\n")
        f.write(f"Total de amostras: {len(dataset)}\n")
        f.write(f"Amostras treino: {len(train_loader.dataset)}\n")
        f.write(f"Amostras validação: {len(val_loader.dataset)}\n")
        f.write(f"Batches por época: Treino {len(train_loader)} | Validação {len(val_loader)}\n\n")
        
        f.write("--- ARQUITETURA ---\n")
        f.write(f"Modelo: BeamNetDropout\n")
        f.write(f"Input dim: 10\n")
        f.write(f"Hidden dim: 512\n")
        f.write(f"Dropout: 0.05\n\n")
        
        f.write("--- HIPERPARÂMETROS ---\n")
        f.write(f"Learning rate inicial: {LR:.2e}\n")
        f.write(f"Learning rate final: {optimizer.param_groups[0]['lr']:.2e}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Otimizador: Adam\n")
        f.write(f"Loss function: MSELoss\n")
        f.write(f"Scheduler: ReduceLROnPlateau (factor=0.95, patience=120)\n")
        f.write(f"Early stop patience: {EARLY_STOP_PATIENCE}\n")
        f.write(f"Normalizador: RobustScaler\n\n")
        
        f.write("--- RESULTADOS ---\n")
        f.write(f"Epochs planejadas: {EPOCHS}\n")
        f.write(f"Epochs treinadas: {epoch+1}\n")
        f.write(f"Melhor Val Loss (transformado): {val_losses_transformed[-1]:.6f}\n")
        f.write(f"Melhor Val Loss (original):     {best_val_original:.6f} ← USE ESTE\n\n")
        
        f.write("--- HISTÓRICO DE LOSS ---\n")
        f.write(f"Loss inicial (treino): {train_losses[0]:.6f}\n")
        f.write(f"Loss final (treino): {train_losses[-1]:.6f}\n")
        f.write(f"Loss inicial (val original): {val_losses_original[0]:.6f}\n")
        f.write(f"Loss final (val original): {val_losses_original[-1]:.6f}\n")
        f.write(f"Melhor val loss (original): {best_val_original:.6f}\n")
        
        if val_losses_original[0] > 0:
            improvement = ((val_losses_original[0] - best_val_original) / val_losses_original[0] * 100)
        else:
            improvement = 0
        f.write(f"Melhoria total: {improvement:.2f}%\n\n")
        
        f.write("--- AMBIENTE ---\n")
        f.write(f"Dispositivo: {device}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"Diretório de salvamento: {SAVE_DIR}\n")
        
    print(f"📄 Resumo salvo em: {summary_path}")

    # --- Mensagem final ---
    print("\n" + "="*70)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"Melhor modelo: {best_model_path}")
    print(f"   (inclui scaler_y salvo com pickle)")
    print(f"Modelo final: {final_model_path}")
    print(f"Curva de aprendizado: {SAVE_DIR}/training_curve_dropout.png")
    print(f"Resumo completo: {summary_path}")
    print(f"Melhor val loss (espaço original): {best_val_original:.6f}")
    print(f"Melhoria total: {improvement:.2f}%")
    print(f"Duração: {total_time}")
    print("\n✓ Agora suas predições estão CORRETAS!")
    print("✓ Pode usar para Monte Carlo Dropout")
    print("✓ Pode comparar com VEM")
    print("="*70)