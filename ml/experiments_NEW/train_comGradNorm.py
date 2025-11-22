import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from utils_data import Beam1DDataset, save_scalers
from models import BeamNet
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Configurações de Caminho Robustas ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATASET_NAME = "dataset_viga1D_EngastadaLivre_1Meter_44Elements_10kSamples"
DATASET_PATH = os.path.join(PROJECT_ROOT, "build", "output", "ML_datasets", DATASET_NAME)
TRAIN_FILE = os.path.join(DATASET_PATH, "training_data.csv")

# --- Configurações ---
EPOCHS = 1000
BATCH_SIZE = 4096
LR_MODEL = 1e-3
LR_WEIGHTS = 1e-2  # Taxa de aprendizado para os pesos do GradNorm
ALPHA = 0.12       # Hiperparâmetro do GradNorm (força de restauração)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"🚀 Iniciando treinamento em: {DEVICE}")

def main():
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
            persistent_workers=True
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

    # 2. Modelo e Otimizadores
    model = BeamNet(input_dim=6, output_dim=2, hidden_dim=256).to(DEVICE)
    
    # Otimizador do Modelo (Pesos da Rede)
    optimizer_model = optim.Adam(model.parameters(), lr=LR_MODEL)
    scheduler = ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=50, min_lr=1e-6)

    # --- SETUP GRADNORM ---
    # Pesos das perdas (w_0, w_1) -> Inicializados em 1.0
    # requires_grad=True é essencial aqui
    loss_weights = torch.ones(2, device=DEVICE, requires_grad=True)
    
    # Otimizador específico para os loss_weights
    optimizer_weights = optim.Adam([loss_weights], lr=LR_WEIGHTS)
    
    # Losses iniciais (para normalização do GradNorm)
    initial_losses = None

    loss_history = {'train': [], 'val': []}

    # 3. Loop de Treino
    for epoch in range(EPOCHS):
        model.train()
        train_loss_epoch = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            # --- 1. Forward Pass ---
            # Acessamos a última camada linear (camada compartilhada de referência)
            last_layer = model.net[-1] 
            
            y_pred = model(X_batch)
            
            # --- 2. Calcular Perdas Individuais ---
            loss_w = torch.mean((y_pred[:, 0] - y_batch[:, 0]) ** 2)
            loss_theta = torch.mean((y_pred[:, 1] - y_batch[:, 1]) ** 2)
            
            losses = torch.stack([loss_w, loss_theta])
            
            # Inicializa perdas base na primeira iteração
            if initial_losses is None:
                initial_losses = losses.detach()
            
            # --- 3. Loss Ponderada (Para atualizar o modelo) ---
            weighted_loss = torch.sum(loss_weights * losses)
            
            optimizer_model.zero_grad()
            # retain_graph=True é necessário pois faremos outro backward depois (para o GradNorm)
            weighted_loss.backward(retain_graph=True) 
            
            # --- 4. GradNorm: Calcular Gradientes das Perdas Ponderadas ---
            # Aqui aplicamos a CORREÇÃO: create_graph=True
            
            shared_layer_params = list(last_layer.parameters()) 
            
            norms = []
            for i in range(2):
                # Calcula gradiente de (w_i * L_i) em relação aos pesos da última camada
                # create_graph=True: Permite derivar este gradiente em relação a loss_weights depois
                grad = torch.autograd.grad(
                    loss_weights[i] * losses[i], 
                    shared_layer_params, 
                    retain_graph=True, 
                    create_graph=True  # <--- A CORREÇÃO CRÍTICA ESTÁ AQUI
                )
                
                # Concatena e calcula norma L2
                grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
                norms.append(grad_norm)
            
            norms = torch.stack(norms)
            
            # --- 5. GradNorm: Calcular Loss para os Pesos (L_grad) ---
            avg_norm = torch.mean(norms)
            
            # Taxa de aprendizado relativa
            # Adicionamos 1e-8 para evitar divisão por zero se a loss inicial for zero (raro, mas seguro)
            loss_ratios = losses.detach() / (initial_losses + 1e-8)
            inverse_train_rates = loss_ratios / torch.mean(loss_ratios)
            
            # Alvo: G_target (trata-se como constante, por isso .detach())
            target_norms = avg_norm * (inverse_train_rates ** ALPHA)
            
            # Loss do GradNorm (L1 Loss)
            grad_loss = torch.abs(norms - target_norms.detach()).sum()
            
            # --- 6. Atualizar Pesos dos Losses ---
            optimizer_weights.zero_grad()
            grad_loss.backward() # Agora isso funciona porque norms tem conexão com loss_weights
            optimizer_weights.step()
            
            # --- 7. Renormalizar Pesos ---
            # Garante que a soma dos pesos continue sendo 2 (número de tarefas)
            with torch.no_grad():
                normalize_coeff = 2.0 / torch.sum(loss_weights)
                loss_weights.data = loss_weights.data * normalize_coeff
            
            # --- 8. Atualizar Modelo ---
            # Finaliza o passo do modelo principal
            optimizer_model.step()
            
            train_loss_epoch += weighted_loss.item()
            
        avg_train_loss = train_loss_epoch / len(train_loader)
        
        # Validação
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                
                l_w = torch.mean((y_pred[:, 0] - y_batch[:, 0]) ** 2)
                l_th = torch.mean((y_pred[:, 1] - y_batch[:, 1]) ** 2)
                
                # Soma ponderada usando os pesos atuais (apenas para métrica)
                val_l = torch.sum(loss_weights.detach() * torch.stack([l_w, l_th]))
                
                val_loss_epoch += val_l.item()
        
        avg_val_loss = val_loss_epoch / len(val_loader)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer_model.param_groups[0]['lr']
        
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        
        if (epoch+1) % 50 == 0:
            w_w, w_theta = loss_weights.tolist()
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.1e}")
            print(f"    ⚖️  GradNorm Weights -> W (Desloc): {w_w:.3f} | Theta (Rot): {w_theta:.3f}")

    # 4. Salvar Modelo
    torch.save(model.state_dict(), f"{DATASET_PATH}/beamnet_model.pth")
    print("✅ Modelo salvo!")

    # 5. Plotar Loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history['train'], label='Treino')
    plt.plot(loss_history['val'], label='Validação')
    plt.yscale('log')
    plt.title('Curva de Convergência (GradNorm)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss Ponderada')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f"{DATASET_PATH}/training_loss.png")
    print("✅ Gráfico de Loss salvo!")

if __name__ == "__main__":
    main()