import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from models import BeamNetDropout
from utils_data import (
    BeamDataset, 
    inverse_transform_predictions,
    get_dataloaders
)

# ============================================================
# 🔹 Configurações
# ============================================================
DATA_DIR = "../build/output/ml_dataset_100k"  # ← Mesmo usado no treino
MODEL_PATH = "experiments/beamnet_dropout_best.pt"  # modelo com dropout
SAVE_DIR = os.path.dirname(MODEL_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 Salvando gráficos em: {SAVE_DIR}")

device = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")


# ============================================================
# 🔹 Carregar modelo E scaler do checkpoint
# ============================================================
print(f"\n📂 Carregando checkpoint de: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# ✅ CORRIGIDO: Carregar modelo do dicionário
model = BeamNetDropout(input_dim=11, hidden_dim=512, dropout_p=0.05)  # ← input_dim=11 (corrigido)
model.load_state_dict(checkpoint["model_state"])  # ← Correto
model.to(device)
model.eval()

# ✅ CORRIGIDO: Carregar scaler_y do checkpoint (com pickle)
scaler_y = pickle.loads(checkpoint["scaler_y_pickle"])  # ← Mesmo scaler do treino
print(f"✓ Modelo carregado")
print(f"✓ Scaler_y carregado (mesmo do treino)")

# Carregar dataset apenas para scaler_x (para normalização de X)
train_loader, val_loader, dataset = get_dataloaders(
    DATA_DIR,
    n_samples=None,
    batch_size=512,
    normalize=True,
    val_split=0.2
)

params = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))


# ============================================================
# 🔹 Função Monte Carlo Dropout (para incerteza)
# ============================================================
def predict_with_uncertainty(model, X_tensor, n_samples=100):
    """
    ✅ CORRIGIDO: Executa múltiplas predições com dropout ativo
    Retorna predições em espaço TRANSFORMADO
    """
    model.train()  # Mantém dropout ativo
    preds = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor)  # Predição em espaço transformado
            preds.append(pred.cpu().numpy())
    
    preds = np.array(preds)  # [n_samples, batch_size, 1]
    
    mean_pred_transformed = preds.mean(axis=0).flatten()  # Média em espaço transformado
    std_pred_transformed = preds.std(axis=0).flatten()    # Std em espaço transformado
    
    return mean_pred_transformed, std_pred_transformed


# ============================================================
# 🔹 Função para inverter incerteza
# ============================================================
def invert_uncertainty(mean_trans, std_trans, scaler_y, n_samples=100):
    """
    ✅ CORRIGIDO: Inverter média e estimar incerteza em espaço original
    
    Para a média: usar inverse_transform
    Para o desvio padrão: usar análise de sensibilidade
    """
    
    # Converter média
    mean_original = inverse_transform_predictions(mean_trans, scaler_y)
    
    # Para o desvio padrão, usar propagação de incerteza
    # Aproximação: calcular em torno da média
    delta = std_trans * 0.01  # Pequeno desvio
    
    # Perturbar em torno da média
    lower_trans = (mean_trans - 2*std_trans).reshape(-1, 1)
    upper_trans = (mean_trans + 2*std_trans).reshape(-1, 1)
    
    lower_orig = inverse_transform_predictions(lower_trans, scaler_y).flatten()
    upper_orig = inverse_transform_predictions(upper_trans, scaler_y).flatten()
    
    # Estimar novo desvio padrão
    std_original = (upper_orig - lower_orig) / 4  # 4 = 2*2 (±2σ)
    
    return mean_original, std_original


# ============================================================
# 🔹 Função para gerar e salvar gráfico de incerteza
# ============================================================
def plot_uncertainty(sample_idx=0, n_mc_samples=100):
    """
    ✅ CORRIGIDO: Gera gráfico com incerteza em espaço ORIGINAL
    """
    print(f"\n📊 Gerando gráfico para amostra {sample_idx}...")
    
    df = pd.read_csv(os.path.join(DATA_DIR, f"sample_{sample_idx:04d}.csv"))
    E, I, q = params.loc[sample_idx, ["E", "I", "q"]]

    # Ordenar por posição x
    df = df.sort_values(by="x")
    x = df["x"].values

    # Carregar y verdadeiro (VEM)
    if "displacement_scaled" in df.columns:
        y_true_transformed = df["displacement_scaled"].values
    else:
        y_true_transformed = df["displacement"].values

    # ✅ CORRIGIDO: Converter y verdadeiro para espaço original
    y_true = inverse_transform_predictions(y_true_transformed, scaler_y).flatten()

    # ---- Montar features (AGORA COM 11 features!) ----
    self_L = 1.0
    x_scaled = x / self_L
    x2, x3, x4 = x**2, x**3, x**4
    
    EI = E * I
    inv_EI = 1.0 / EI
    q_over_EI = q / EI
    theoretical_disp = -(q * x**2 * (6*self_L**2 - 4*self_L*x + x**2)) / (24 * E * I)

    X_in = np.stack([
        np.log10(E) * np.ones_like(x),
        np.log10(I) * np.ones_like(x),
        np.log10(abs(q) + 1e-9) * np.ones_like(x),
        x_scaled, x2, x3, x4,
        EI * np.ones_like(x),
        inv_EI * np.ones_like(x),
        q_over_EI * np.ones_like(x),
        theoretical_disp
    ], axis=1)

    # ---- Normalizar X ----
    X_norm = dataset.scaler_x.transform(X_in)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    # ---- Predição com incerteza (Monte Carlo Dropout) ----
    mean_pred_trans, std_pred_trans = predict_with_uncertainty(
        model, X_tensor, n_samples=n_mc_samples
    )

    # ✅ CORRIGIDO: Inverter para espaço original
    mean_pred, std_pred = invert_uncertainty(
        mean_pred_trans, std_pred_trans, scaler_y
    )

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, "k-", linewidth=2.5, label="VEM (referência)")
    plt.plot(x, mean_pred, "r--", linewidth=2, label="NN (predição média)")
    plt.fill_between(
        x,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        color="orange",
        alpha=0.3,
        label="±2σ (incerteza)"
    )
    plt.xlabel("Posição x (m)", fontsize=12)
    plt.ylabel("Deslocamento (m)", fontsize=12)
    plt.title(f"Monte Carlo Dropout - Amostra {sample_idx}\n(n_samples={n_mc_samples})", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()

    # ---- Salvar ----
    save_path = os.path.join(SAVE_DIR, f"uncertainty_sample_{sample_idx:04d}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"   ✓ Gráfico salvo: {save_path}")
    print(f"   Erro médio (espaço original): {np.mean(np.abs(mean_pred - y_true)):.6f} m")
    print(f"   Desvio padrão médio: {np.mean(std_pred):.6f} m")


# ============================================================
# 🔹 Rodar para múltiplas vigas
# ============================================================
print("\n🚀 Gerando gráficos de incerteza...\n")
for idx in [0, 10, 25, 50, 75]:
    try:
        plot_uncertainty(sample_idx=idx, n_mc_samples=100)
    except Exception as e:
        print(f"   ⚠️  Erro ao processar amostra {idx}: {e}")

print("\n✅ Todos os gráficos foram gerados!")