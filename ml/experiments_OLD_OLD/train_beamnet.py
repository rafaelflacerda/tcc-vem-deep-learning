import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils_data import get_dataloaders
from models import BeamNet


# ============================================================
# 🔹 Função de treino e validação
# ============================================================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Treino {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)


def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(val_loader.dataset)


if __name__ == "__main__":
    # ============================================================
    # 🔹 Configurações gerais
    # ============================================================
    DATA_DIR = "../build/output/ml_dataset"    # caminho do dataset gerado pelo beam1d
    SAVE_DIR = "experiments"              # onde salvar modelos e gráficos
    EPOCHS = 1200
    BATCH_SIZE = 512
    LR = 1e-4

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Detectar hardware
    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")


    # ============================================================
    # 🔹 Carregar dataset
    # ============================================================
    print("\n📦 Carregando dataset...")
    train_loader, val_loader, dataset = get_dataloaders(
        DATA_DIR,
        n_samples=None,          # usa todas as 10.000 amostras
        batch_size=BATCH_SIZE,
        normalize=True,
        val_split=0.1
    )
    print(f"✔️ Dados carregados: {len(dataset)} amostras totais")
    print(f"✔️ Batches de treino: {len(train_loader)} | Validação: {len(val_loader)}")


    # ============================================================
    # 🔹 Instanciar modelo
    # ============================================================
    model = BeamNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)


    # ============================================================
    # 🔹 Loop principal de treinamento
    # ============================================================
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_model_path = os.path.join(SAVE_DIR, "beamnet_best.pt")

    print("\n🎯 Iniciando treinamento...\n")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(epoch)
        val_loss = validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"📉 Época {epoch+1:03d} | "
              f"Treino: {train_loss:.6f} | Validação: {val_loss:.6f}")

        # Salvar melhor modelo
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_model_path)

    # ============================================================
    # 🔹 Plotar curva de aprendizado
    # ============================================================
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.yscale("log")
    plt.xlabel('Época')
    plt.ylabel('MSE Loss')
    plt.title('Curva de aprendizado - BeamNet')
    plt.legend()
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), dpi=150)
    plt.show()


# ============================================================
# 🔹 Salvar modelo final e parâmetros de normalização (novo formato)
# ============================================================
final_model_path = os.path.join(SAVE_DIR, "beamnet_final.pt")

torch.save({
    "model_state": model.state_dict(),
    "scaler_x_mean": dataset.scaler_x.mean_.tolist(),
    "scaler_x_scale": dataset.scaler_x.scale_.tolist(),
    "scaler_y_mean": dataset.scaler_y.mean_.tolist(),
    "scaler_y_scale": dataset.scaler_y.scale_.tolist(),
}, final_model_path)

print("\n✅ Treinamento concluído!")
print(f"📁 Melhor modelo salvo em: {best_model_path}")
print(f"📁 Modelo final salvo em: {final_model_path}")
print(f"📈 Curva de aprendizado salva em: {SAVE_DIR}/training_curve.png")