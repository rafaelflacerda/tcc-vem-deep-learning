import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class OptimizedBeamDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        
        # Carregar tudo para RAM de uma vez
        X = data['X'].astype(np.float32)
        y = data['Y'].astype(np.float32)
        
        # Normalização
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)
        
        X = (X - self.X_mean) / (self.X_std + 1e-8)
        y = (y - self.y_mean) / (self.y_std + 1e-8)
        
        # Converter para tensors de uma vez (GPU)
        self.X = torch.tensor(X, dtype=torch.bfloat16)
        self.y = torch.tensor(y, dtype=torch.bfloat16)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PreprocessedBeamDataset(Dataset):
    def __init__(self, npz_path, scalers_path=None):
        data = np.load(npz_path)

        # X, Y já normalizados no preprocess
        X = data["X"]    # float32
        y = data["Y"]    # float32

        # Guarda scalers se quiser usar depois (opcional)
        if scalers_path is not None:
            s = np.load(scalers_path)
            self.X_mean = s["X_mean"]
            self.X_std  = s["X_std"]
            self.y_mean = s["y_mean"]
            self.y_std  = s["y_std"]
        else:
            self.X_mean = self.X_std = self.y_mean = self.y_std = None

        # Converte para tensor uma vez
        self.X = torch.tensor(X, dtype=torch.bfloat16)
        self.y = torch.tensor(y, dtype=torch.bfloat16)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_optimized_loader(dataset, batch_size, shuffle=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Dados já em RAM/GPU
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False
    )

def save_scalers(X_mean, X_std, y_mean, y_std, output_path):
    """
    Salva os scalers usados na normalização em um arquivo .npz.

    Args:
        X_mean, X_std, y_mean, y_std: arrays numpy com médias e desvios
        output_path: caminho do arquivo .npz (ex: "outputs/scalers_beam_250k.npz")
    """
    scalers = {
        "X_mean": np.asarray(X_mean, dtype=np.float32),
        "X_std":  np.asarray(X_std,  dtype=np.float32),
        "y_mean": np.asarray(y_mean, dtype=np.float32),
        "y_std":  np.asarray(y_std,  dtype=np.float32),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **scalers)

def load_scalers(path="scalers.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["scaler_X"], data["scaler_y"]