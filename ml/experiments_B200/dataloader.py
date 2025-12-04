import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class OptimizedBeamDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        
        # Carregar tudo para RAM de uma vez
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)
        
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