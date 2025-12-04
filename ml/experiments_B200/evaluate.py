import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_data import Beam1DDataset, load_scalers
from models import BeamNet
from physics import analytical_solution

# --- Configurações de Caminho Robustas ---
# Pega o diretório onde este script está localizado (ml/experiments_NEW)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Volta duas pastas para chegar na raiz do projeto (tcc-vem-deep-learning)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Define o caminho absoluto para o dataset
DATASET_NAME = "dataset_viga1D_EngastadaLivre_1Meter_44Elements_100kSamples"
DATASET_PATH = os.path.join(PROJECT_ROOT, "build", "output", "ML_datasets", DATASET_NAME)

# Define os arquivos dentro da pasta do dataset
VAL_FILE = os.path.join(DATASET_PATH, "full_validation_data.csv")
MODEL_PATH = os.path.join(DATASET_PATH, "beamnet_model.pth")
SCALER_PATH = os.path.join(DATASET_PATH, "scalers.pkl")

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"🚀 Avaliando em: {DEVICE.upper()}")

# Verificação de segurança
if not os.path.exists(VAL_FILE):
    raise FileNotFoundError(f"❌ Erro Crítico: Arquivo de validação não encontrado em:\n{VAL_FILE}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Erro Crítico: Modelo treinado não encontrado. Rode o train.py primeiro.\nCaminho esperado: {MODEL_PATH}")

def evaluate_single_beam(sample_id=0):
    print(f"\n🔍 Analisando amostra ID: {sample_id}...")

    # 1. Carregar Scalers e Modelo
    try:
        scaler_X, scaler_y = load_scalers(SCALER_PATH)
        model = BeamNet(input_dim=6, hidden_dim=512).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Erro ao carregar modelo/scalers: {e}")
        return
    
    # 2. Carregar dados da amostra específica
    df = pd.read_csv(VAL_FILE)
    beam_data = df[df['sample_id'] == sample_id].sort_values('x_position')
    
    if len(beam_data) == 0:
        print(f"⚠️  ID de amostra {sample_id} não encontrado no CSV.")
        return

    # 3. Preparar entrada para a rede (Viés Físico)
    X_input = []
    # Pega os parâmetros da primeira linha da amostra (são constantes para a viga toda)
    E = beam_data.iloc[0]['E']
    I = beam_data.iloc[0]['I']
    q = beam_data.iloc[0]['q']
    
    E_log = np.log10(E)
    I_log = np.log10(I)
    
    print(f"   Parâmetros: E={E:.2e}, I={I:.2e}, q={q:.2f}")
    
    x_vals = beam_data['x_position'].values
    y_vem_w = beam_data['w'].values
    
    # Recalcula inputs analíticos para cada ponto x
    for x in x_vals:
        w_ana, th_ana = analytical_solution(E, I, q, x)
        X_input.append([x, E_log, I_log, q, w_ana, th_ana])
    
    X_input = np.array(X_input)
    
    # Normalizar
    X_scaled = scaler_X.transform(X_input)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    
    # 4. Predição (com Incerteza via MC Dropout)
    model.train() # Ativa dropout para estimar incerteza
    n_samples = 100
    predictions = []
    
    print("   🎲 Executando Monte Carlo Dropout...")
    with torch.no_grad():
        for _ in range(n_samples):
            pred_scaled = model(X_tensor).cpu().numpy()
            pred_real = scaler_y.inverse_transform(pred_scaled)
            predictions.append(pred_real)
            
    predictions = np.array(predictions) # Shape: [100, 25, 2]
    
    # Média e Desvio Padrão das predições
    mean_preds = predictions.mean(axis=0)
    std_preds = predictions.std(axis=0)
    
    w_pred = mean_preds[:, 0] # Coluna 0 é deflexão (w)
    w_std = std_preds[:, 0]
    
    # 5. Plotagem
    plt.figure(figsize=(10, 6))
    
    # VEM (Ground Truth)
    plt.plot(x_vals, y_vem_w, 'k-', linewidth=2, label='VEM (Target)')
    
    # Rede Neural
    plt.plot(x_vals, w_pred, 'r--', label='Rede Neural (Média)')
    plt.fill_between(x_vals, w_pred - 2*w_std, w_pred + 2*w_std, color='red', alpha=0.2, label='Incerteza (95% conf)')
    
    # Solução Analítica (Viés)
    # A coluna 4 do X_input contém o w_analitico que entrou na rede
    w_pure_ana = -X_input[:, 4] 
    plt.plot(x_vals, w_pure_ana, 'g:', alpha=0.5, label='Solução Analítica (Input)')
    
    plt.title(f'Validação Viga 1D - Amostra {sample_id}')
    plt.xlabel('Posição x (m)')
    plt.ylabel('Deslocamento w (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot_path = os.path.join(DATASET_PATH, f"validation_sample_{sample_id}.png")
    plt.savefig(save_plot_path)
    print(f"✅ Gráfico salvo em: {save_plot_path}")
    # plt.show() # Comente se estiver rodando em servidor sem tela

if __name__ == "__main__":
    # Avalia algumas amostras para garantir que não foi "sorte"
    evaluate_single_beam(sample_id=42) 
    evaluate_single_beam(sample_id=100)
    evaluate_single_beam(sample_id=500)