import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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