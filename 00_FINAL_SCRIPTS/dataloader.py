import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(npz_path, batch_size=4096, num_workers=8, train_split=0.8, seed=42):
    """
    Carrega NPZ e retorna train/val DataLoaders
    Split por casos (não por nós) para evitar data leakage
    """
    # Carregar dados
    data = np.load(npz_path)
    X = data['X']
    Y = data['Y']
    case_indices = data['case_indices']
    
    # Casos únicos
    unique_cases = np.unique(case_indices)
    
    # Shuffle com seed fixo (reprodutibilidade)
    np.random.seed(seed)
    np.random.shuffle(unique_cases)
    
    # Split 80/20
    n_train = int(train_split * len(unique_cases))
    train_cases = unique_cases[:n_train]
    val_cases = unique_cases[n_train:]
    
    # Filtrar nós por caso
    train_mask = np.isin(case_indices, train_cases)
    val_mask = np.isin(case_indices, val_cases)
    
    X_train = torch.tensor(X[train_mask], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_mask], dtype=torch.float32)
    X_val = torch.tensor(X[val_mask], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_mask], dtype=torch.float32)
    
    # Datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Acelera transferência CPU→GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_cases)} casos ({len(X_train)} nós)")
    print(f"Val: {len(val_cases)} casos ({len(X_val)} nós)")
    
    return train_loader, val_loader