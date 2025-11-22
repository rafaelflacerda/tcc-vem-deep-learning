import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import BeamNet
from utils_data import BeamDataset

# ============================================================
# 🔹 Configurações
# ============================================================
DATA_DIR = "../build/output/ml_dataset"  # mesma pasta usada no treino
MODEL_PATH = "experiments/beamnet_best.pt"

device = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")


# ============================================================
# 🔹 Carregar modelo e normalização
# ============================================================
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = BeamNet()
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# ============================================================
# 🔹 Carregar dataset
# ============================================================
dataset = BeamDataset(DATA_DIR, n_samples=100, normalize=True)  # apenas 100 amostras p/ plotar
# Extrair parâmetros de normalização do StandardScaler
X_mean = dataset.scaler_x.mean_
X_std  = dataset.scaler_x.scale_
y_mean = dataset.scaler_y.mean_
y_std  = dataset.scaler_y.scale_

# ============================================================
# 🔹 Selecionar uma viga e reconstruir curva real vs predita
# ============================================================
def plot_prediction(sample_idx=0):
    params = pd.read_csv(os.path.join(DATA_DIR, "parameters.csv"))
    df = pd.read_csv(os.path.join(DATA_DIR, f"sample_{sample_idx:04d}.csv"))

    E, I, q = params.loc[sample_idx, ["E", "I", "q"]]
    x = df["x"].values
    y_true = df["displacement"].values

    # 🔹 Adicionar o termo quadrático de x e o fator de escala
    x_scaled = 5.0 * x
    x2 = x ** 2
    x3 = x ** 3
    x4 = x ** 4

    X_in = np.stack([
        np.log10(E) * np.ones_like(x),
        np.log10(I) * np.ones_like(x),
        np.log10(abs(q) + 1e-9) * np.ones_like(x),
        x_scaled,
        x2,
        x3,
        x4
    ], axis=1)

    # --- Normalizar com os scalers do dataset ---
    X_norm = dataset.scaler_x.transform(X_in)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)

    # --- Inferência ---
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()

    # --- Desnormalizar saída ---
    if hasattr(dataset, "scaler_y"):
        y_pred = dataset.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_true = dataset.scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()

    # --- Plot ---
    plt.figure(figsize=(7,5))
    plt.plot(x, y_true, "k-", label="VEM (real)")
    plt.plot(x, y_pred, "r--", label="Rede Neural")
    plt.xlabel("Posição x (m)")
    plt.ylabel("Deslocamento (m)")
    plt.title(f"Comparação VEM × NN (sample {sample_idx})")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    # Salvar gráfico automaticamente na pasta experiments/
    save_path = os.path.join("experiments", f"comparison_sample_{sample_idx:04d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📈 Gráfico salvo: {save_path}")

    # --- Métrica de erro ---
    mse = np.mean((y_pred - y_true) ** 2)
    print(f"MSE da viga {sample_idx:04d}: {mse:.4e}")


# ============================================================
# 🔹 Avaliar múltiplas vigas
# ============================================================
for idx in [0, 10, 25, 50, 75]:
    plot_prediction(sample_idx=idx)