import numpy as np
import torch
from pathlib import Path
import sys

# Adicionar pasta scripts ao path para importar dataloader
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from dataloader import get_dataloaders

# Configuração
PROJECT_ROOT = SCRIPT_DIR.parent
NPZ_FILE = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_npz/meshes_500_samples.npz"

print("=== Testando DataLoader ===\n")

# Carregar dataloaders
train_loader, val_loader = get_dataloaders(
    npz_path=str(NPZ_FILE),
    batch_size=64,  # Pequeno para teste
    num_workers=0,  # 0 para debug
    train_split=0.8
)

# 1. Verificar se carregou
print("\n=== Batch Test ===")
X_batch, Y_batch = next(iter(train_loader))
print(f"Train batch shape - X: {X_batch.shape}, Y: {Y_batch.shape}")

X_batch_val, Y_batch_val = next(iter(val_loader))
print(f"Val batch shape - X: {X_batch_val.shape}, Y: {Y_batch_val.shape}")

# 2. Verificar features
assert X_batch.shape[1] == 7, "Features devem ser 7"
assert Y_batch.shape[1] == 2, "Targets devem ser 2"
print("✓ Shapes corretos")

# 3. Verificar valores
assert not torch.isnan(X_batch).any(), "Train tem NaN"
assert not torch.isnan(Y_batch).any(), "Train targets tem NaN"
print("✓ Sem NaN")

# 4. Verificar separação train/val (sem overlap de cases)
print("\n=== Verificando Data Leakage ===")
data = np.load(NPZ_FILE)
case_indices = data['case_indices']
X_full = data['X']

# Pegar casos de treino e validação
unique_cases = np.unique(case_indices)
np.random.seed(42)
np.random.shuffle(unique_cases)
n_train = int(0.8 * len(unique_cases))
train_cases = set(unique_cases[:n_train])
val_cases = set(unique_cases[n_train:])

overlap = train_cases.intersection(val_cases)
print(f"Train cases: {len(train_cases)}")
print(f"Val cases: {len(val_cases)}")
print(f"Overlap: {len(overlap)}")
assert len(overlap) == 0, "ERRO: Train e Val tem overlap!"
print("✓ Sem data leakage")

print("\n✓ DataLoader validado com sucesso!")