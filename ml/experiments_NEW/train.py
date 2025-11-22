import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from utils_data import Beam1DDataset, save_scalers
from models import BeamNet
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configurações de Caminho Robustas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Caminho para o dataset
DATASET_NAME = "dataset_viga1D_EngastadaLivre_1Meter_44Elements_100kSamples"
DATASET_PATH = os.path.join(PROJECT_ROOT, "build", "output", "ML_datasets", DATASET_NAME)
TRAIN_FILE = os.path.join(DATASET_PATH, "training_data.csv")

# --- Configurações ---
EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-3
LAMBDA_THETA = 10.0  # ✅ Peso da Derivada (Sobolev): Aumente se a rotação estiver imprecisa

# --- Configuração de Hardware ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍏 Apple Silicon detectado! Usando Metal Performance Shaders (GPU).")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🔥 GPU NVIDIA detectada! Usando CUDA.")
else:
    DEVICE = "cpu"
    print("⚠️  Nenhuma GPU aceleradora detectada. Usando CPU.")

# ============================================================
# ✅ NOVA FUNÇÃO DE PERDA: SOBOLEV DE 1ª ORDEM
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
    print(f"🚀 Iniciando treinamento em: {DEVICE}")
    print(f"⚖️  Usando Sobolev Loss com Lambda_Theta = {LAMBDA_THETA}")
    
    # 1. Carregar Dados
    full_dataset = Beam1DDataset(TRAIN_FILE, is_training=True)
    save_scalers(full_dataset.scaler_X, full_dataset.scaler_y, f"{DATASET_PATH}/scalers.pkl")
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    #train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,        # Usa 8 núcleos do M4
        persistent_workers=True # Mantém processos vivos
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=8,
        persistent_workers=True
    )
    
    print(f"📊 Dados: {len(full_dataset)} pontos totais")
    print(f"   Treino: {train_size} | Validação Interna: {val_size}")

    # 2. Modelo e Otimizador
    model = BeamNet(input_dim=6, output_dim=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, min_lr = 1e-6)
    
    # ⚠️  Nota: Removemos 'criterion = nn.MSELoss()' aqui. 
    # Usaremos 'sobolev_loss' diretamente no loop.
    
    loss_history = {'train': [], 'val': []}

    # 3. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            # ✅ CHAMADA SOBOLEV NO TREINO
            loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                
                # ✅ CHAMADA SOBOLEV NA VALIDAÇÃO (Para métrica consistente)
                loss = sobolev_loss(y_pred, y_batch, lambda_theta=LAMBDA_THETA)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Obter LR atual para log/print
        current_lr = optimizer.param_groups[0]['lr']
        
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e} ")

    # 4. Salvar Modelo
    torch.save(model.state_dict(), f"{DATASET_PATH}/beamnet_model.pth")
    print("✅ Modelo salvo!")

    # 5. Plotar Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='Treino')
    plt.plot(loss_history['val'], label='Validação')
    plt.yscale('log')
    plt.title(f'Curva de Convergência (Sobolev Lambda={LAMBDA_THETA})')
    plt.xlabel('Épocas')
    plt.ylabel('Loss Ponderada')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"{DATASET_PATH}/training_loss.png")
    print("✅ Gráfico de Loss salvo!")

if __name__ == "__main__":
    main()