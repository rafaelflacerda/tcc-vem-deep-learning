import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Caminhos relativos ao script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuração
JSON_DIR = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_json/meshes_1000_samples"
OUTPUT_FILE = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_npz/meshes_1000_samples.npz"

# Criar pasta de saída se não existir
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Listas para acumular
all_X = []
all_Y = []
all_case_ids = []

json_files = sorted(JSON_DIR.glob("*.json"))

for jfile in tqdm(json_files):
    with open(jfile) as f:
        data = json.load(f)
    
    case_id = data['case_id']
    L = data['L']
    H = data['H']
    R = data['R']
    nu = data['poisson']
    
    nodes = np.array(data['nodes'])
    solution_flat = np.array(data['solution_u'])
    solution = solution_flat.reshape(-1, 2)
    
    # Vetorização
    x = nodes[:, 0]
    y = nodes[:, 1]
    d = np.sqrt(x**2 + y**2) - R
    
    N = len(nodes)
    L_col = np.full(N, L)
    H_col = np.full(N, H)
    R_col = np.full(N, R)
    nu_col = np.full(N, nu)
    
    X = np.column_stack([x, y, d, L_col, H_col, R_col, nu_col])
    Y = solution
    case_ids = np.full(N, case_id)
    
    all_X.append(X)
    all_Y.append(Y)
    all_case_ids.append(case_ids)

# Concatenar
X = np.vstack(all_X).astype(np.float32)
Y = np.vstack(all_Y).astype(np.float32)
case_indices = np.concatenate(all_case_ids).astype(np.int32)

# Normalizar (com epsilon)
X_max = np.max(np.abs(X[:, :6]), axis=0)
X[:, :6] = X[:, :6] / (X_max + 1e-8)

# Salvar
np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    Y=Y,
    case_indices=case_indices,
    X_max=X_max
)

print(f"Salvo: {X.shape[0]} nós, {len(np.unique(case_indices))} casos")