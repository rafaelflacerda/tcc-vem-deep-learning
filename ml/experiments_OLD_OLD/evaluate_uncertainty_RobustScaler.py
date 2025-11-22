import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from models import BeamNetDropout
from utils_data import BeamDataset, inverse_transform_predictions

# ============================================================
# 🔹 Configurações
# ============================================================
DATA_DIR = "../build/output/ml_dataset_100k"  # ← MESMO DO TREINO
MODEL_PATH = "experiments_RobustScaler/beamnet_dropout_best.pt"
SAVE_DIR = os.path.dirname(MODEL_PATH)
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"📁 Salvando gráficos em: {SAVE_DIR}")

device = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")


# ============================================================
# 🔹 Carregar modelo e scaler_y do checkpoint
# ============================================================
print(f"\n📂 Carregando checkpoint de: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = BeamNetDropout(input_dim=10, hidden_dim=512, dropout_p=0.05)
model.to(device)
model.eval()

# ✅ CORRIGIDO: Detectar formato e carregar scaler_y do checkpoint
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    print("✓ Formato novo detectado (com save_checkpoint)")
    model.load_state_dict(checkpoint["model_state"])
    scaler_y = pickle.loads(checkpoint["scaler_y_pickle"])  # ← CARREGA DO CHECKPOINT
    print("✓ Scaler_y carregado do checkpoint com pickle")
else:
    # Formato antigo
    print("✓ Formato antigo detectado (apenas state_dict)")
    model.load_state_dict(checkpoint)
    scaler_y = None
    print("⚠️  Scaler_y não encontrado no checkpoint, será criado novo")

# Carregar dataset para pegar scaler_x
dataset = BeamDataset(DATA_DIR, n_samples=100, normalize=True)
scaler_x = dataset.scaler_x

# Se não tem scaler_y no checkpoint, usa do dataset
if scaler_y is None:
    scaler_y = dataset.scaler_y
    print("   Usando scaler_y do dataset")

params = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))


# ============================================================
# 🔹 Função Monte Carlo Dropout (para incerteza)
# ============================================================
def predict_with_uncertainty(model, X_tensor, n_samples=100):
    """
    ✅ Executa múltiplas predições com dropout ativo (Monte Carlo)
    Retorna predições em espaço TRANSFORMADO
    """
    model.train()  # Mantém dropout ativo
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor)
            preds.append(pred.cpu().numpy())
    preds = np.array(preds)
    
    mean_pred_transformed = preds.mean(axis=0).flatten()
    std_pred_transformed = preds.std(axis=0).flatten()
    return mean_pred_transformed, std_pred_transformed


# ============================================================
# 🔹 Função para inverter incerteza
# ============================================================
def invert_uncertainty_mc(mean_trans, std_trans, scaler_y):
    """
    ✅ Inverter média e estimar incerteza em espaço original
    """
    # Converter média
    mean_original = inverse_transform_predictions(mean_trans, scaler_y)
    
    # Para o desvio padrão, usar propagação de incerteza
    lower_trans = (mean_trans - 2*std_trans).reshape(-1, 1)
    upper_trans = (mean_trans + 2*std_trans).reshape(-1, 1)
    
    lower_orig = inverse_transform_predictions(lower_trans, scaler_y).flatten()
    upper_orig = inverse_transform_predictions(upper_trans, scaler_y).flatten()
    
    # Estimar novo desvio padrão
    std_original = (upper_orig - lower_orig) / 4
    
    return mean_original, std_original


# ============================================================
# 🔹 Função para gerar e salvar gráfico de incerteza
# ============================================================
def plot_uncertainty(sample_idx=0, n_mc_samples=100):
    """
    ✅ Gera gráfico com incerteza em espaço ORIGINAL
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

    # ---- Montar features (10 features) ----
    L = 1.0
    x_scaled = x / L
    x2 = x_scaled ** 2
    x3 = x_scaled ** 3
    x4 = x_scaled ** 4

    EI = E * I
    inv_EI = 1.0 / EI
    q_over_EI = q / EI
    EI_log = np.log10(EI)
    q_log = np.log10(abs(q) + 1e-9)
    theoretical_disp = -(q * x**2 * (6*L**2 - 4*L*x + x**2)) / (24 * EI)

    X_in = np.stack([
        EI_log * np.ones_like(x),
        q_log * np.ones_like(x),
        x_scaled, x2, x3, x4,
        EI * np.ones_like(x),
        inv_EI * np.ones_like(x),
        q_over_EI * np.ones_like(x),
        theoretical_disp
    ], axis=1)

    # ---- Normalizar X ----
    X_norm = scaler_x.transform(X_in)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    # ---- Predição com incerteza (Monte Carlo Dropout) ----
    mean_pred_trans, std_pred_trans = predict_with_uncertainty(
        model, X_tensor, n_samples=n_mc_samples
    )

    # ✅ CORRIGIDO: Inverter para espaço original
    mean_pred, std_pred = invert_uncertainty_mc(
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
    print(f"   Erro médio: {np.mean(np.abs(mean_pred - y_true)):.6f} m")
    print(f"   Incerteza média (±2σ): {np.mean(std_pred*2):.6f} m")


# ============================================================
# 🔹 Rodar para múltiplas vigas
# ============================================================
print("\n🚀 Gerando gráficos de incerteza com Monte Carlo Dropout...\n")
for idx in [0, 10, 25, 50, 75]:
    try:
        plot_uncertainty(sample_idx=idx, n_mc_samples=100)
    except Exception as e:
        print(f"   ⚠️  Erro ao processar amostra {idx}: {e}")

print("\n✅ Todos os gráficos foram gerados!")