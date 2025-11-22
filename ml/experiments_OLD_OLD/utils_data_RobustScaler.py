import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler

# ============================================================
# 🔹 Dataset personalizado para o problema da viga (beam1d)
# ============================================================
class BeamDataset(Dataset):
    def __init__(self, data_dir, n_samples=None, normalize=True, random_seed = 42):
        """
        data_dir: caminho da pasta onde estão os CSVs
        n_samples: número de amostras a carregar (None = todas)
        normalize: se True, normaliza entradas e saídas
        """
        np.random.seed(random_seed)
        self.data_dir = data_dir
        self.params = pd.read_csv(os.path.join(data_dir, "parameters.csv"))

        # Se quiser limitar o número de amostras (para testes rápidos)
        if n_samples is not None:
            self.params = self.params.iloc[:n_samples]

        # --- Carregar dados ---
        X, y = [], []
        for i in range(len(self.params)):
            sample_file = os.path.join(data_dir, f"sample_{i:04d}.csv")
            if not os.path.exists(sample_file):
                continue

            df = pd.read_csv(sample_file)
            
            df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            
            if "displacement_scaled" in df.columns:
                mean_disp = df["displacement_scaled"].mean()
                std_disp = df["displacement_scaled"].std()
                df = df[(np.abs(df["displacement_scaled"] - mean_disp) <= 3 * std_disp)]
            
            E = self.params.loc[i, "E"]
            I = self.params.loc[i, "I"]
            q = self.params.loc[i, "q"]

            # Para cada ponto ao longo da viga
            for _, row in df.iterrows():
                x_pos = row["x"]
                disp = row["displacement_scaled"]
                
                # Escalonamento logarítmico das entradas físicas
                #E_log = np.log10(E)
                #I_log = np.log10(I)
                EI_log = np.log10(E * I)
                q_log = np.log10(abs(q) + 1e-9)  # evitar log(0)
                
                #x_scaled = 5.0 * x_pos
                self.L = 1.0
                x_scaled = x_pos / self.L
                x2 = x_pos ** 2
                x3 = x_pos ** 3
                x4 = x_pos ** 4
                
                EI = E * I
                inv_EI = 1.0 / EI
                q_over_EI = q / EI
                theoretical_disp = -(q * x_pos**2 * (6*self.L**2 - 4*self.L*x_pos + x_pos**2)) / (24 * EI)
                
                #X.append([E_log, I_log, q_log, x_scaled, x2, x3, x4])
                X.append([
                EI_log, q_log,         # variáveis físicas originais (log)
                x_scaled, x2, x3, x4,        # posição polinomial adimensional
                EI, inv_EI, q_over_EI,       # combinações físicas diretas
                theoretical_disp              # deflexão teórica correta (viés físico)
                ])
                
                y.append(disp)

        self.X = np.array(X, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64).reshape(-1, 1)

        # --- Normalização ---
        self.normalize = normalize
        #if normalize:
            #self.scaler_x = StandardScaler()
            #self.scaler_y = StandardScaler()
            #self.X = self.scaler_x.fit_transform(self.X)
            #self.y = self.scaler_y.fit_transform(self.y)
        if normalize:
             # RobustScaler para normalização robusta contra outliers
            self.scaler_x = RobustScaler()
            self.scaler_y = RobustScaler()
            
            self.X = self.scaler_x.fit_transform(self.X)
            self.y = self.scaler_y.fit_transform(self.y)   
        else:
            self.X_mean = self.X_std = self.y_mean = self.y_std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================
# 🔹 Função auxiliar para criar dataloaders (treino/validação)
# ============================================================
def get_dataloaders(data_dir, n_samples=None, batch_size=512, normalize=True, val_split=0.2):
    """
    Cria dataloaders PyTorch prontos para treino e validação.
    """
    dataset = BeamDataset(data_dir, n_samples, normalize, random_seed=42)

    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    generator = torch.Generator()
    generator.manual_seed(42)

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, dataset

# ============================================================
# ✅ FUNÇÕES PARA INVERSE TRANSFORM (RobustScaler)
# ============================================================

def inverse_transform_predictions(predictions, scaler_y):
    """
    ✅ Converte predições de espaço transformado → espaço original
    
    Args:
        predictions: np.ndarray ou torch.Tensor em espaço transformado
        scaler_y: RobustScaler fitted
    
    Returns:
        np.ndarray em espaço original (físico)
    """
    
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.detach().cpu().numpy()
    else:
        pred_np = predictions
    
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(-1, 1)
    
    # RobustScaler TEM inverse_transform!
    pred_original = scaler_y.inverse_transform(pred_np)
    
    return pred_original


def validate_with_inverse(model, val_loader, criterion, device, scaler_y, verbose=True):
    """
    ✅ Validação que mostra erro tanto em espaço transformado quanto original
    """
    model.eval()
    total_loss_transformed = 0
    total_loss_original = 0
    
    with torch.no_grad():
        for xb, yb_transformed in val_loader:
            xb, yb_transformed = xb.to(device), yb_transformed.to(device)
            
            pred_transformed = model(xb)
            loss_transformed = criterion(pred_transformed, yb_transformed)
            
            yb_original = inverse_transform_predictions(yb_transformed, scaler_y)
            pred_original = inverse_transform_predictions(pred_transformed, scaler_y)
            
            loss_original = np.mean((pred_original - yb_original) ** 2)
            
            total_loss_transformed += loss_transformed.item() * xb.size(0)
            total_loss_original += loss_original * xb.size(0)
    
    avg_loss_transformed = total_loss_transformed / len(val_loader.dataset)
    avg_loss_original = total_loss_original / len(val_loader.dataset)
    
    if verbose:
        print(f"     Val Loss (transformed): {avg_loss_transformed:.6f}")
        print(f"     Val Loss (original):    {avg_loss_original:.6f} ← USE ESTE")
    
    return avg_loss_transformed, avg_loss_original


def save_checkpoint(model, optimizer, scaler_y, epoch, loss, save_path):
    """✅ Salva modelo COM scaler_y completo"""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "scaler_y_pickle": pickle.dumps(scaler_y),
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint salvo em: {save_path}")


def load_checkpoint(model, optimizer, load_path, device):
    """✅ Carrega modelo COM scaler_y completo"""
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler_y = pickle.loads(checkpoint["scaler_y_pickle"])
    
    print(f"✓ Checkpoint carregado de: {load_path}")
    print(f"  Época: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")
    
    return model, optimizer, scaler_y, checkpoint["epoch"]