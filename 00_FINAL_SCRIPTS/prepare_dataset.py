import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Caminhos relativos ao script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Configuração
JSON_DIR = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_json_sigma/meshes_5_samples"
OUTPUT_FILE = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_npz/meshes_5_samples.npz"

# Criar pasta de saída se não existir
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Listas para acumular (nó a nó)
all_X = []
all_Y = []
all_case_ids = []

# listas para guardar nós brutos por caso
all_nodes_raw = []          # (por caso) matriz (N_case, 2)
all_nodes_case_ids = []     # (por nó) case_id de cada nó

# listas para guardar elements por caso
all_elements = []               # (por caso) matriz (ne_case, 3)
all_elements_case_ids = []      # (por elemento) case_id de cada element

# Listas para acumular (element-wise, para tensões)
all_sigma = []
all_sigma_case_ids = []

json_files = sorted(JSON_DIR.glob("*.json"))

for jfile in tqdm(json_files):
    with open(jfile) as f:
        data = json.load(f)
    
    case_id = data['case_id']
    L = data['L']
    H = data['H']
    R = data['R']
    nu = data['poisson']
    
    nodes = np.array(data['nodes'], dtype=np.float32)
    solution_flat = np.array(data['solution_u'], dtype=np.float32)
    solution = solution_flat.reshape(-1, 2)  # (N, 2)
    
    # Vetorização nodal
    x = nodes[:, 0]
    y = nodes[:, 1]
    d = np.sqrt(x**2 + y**2) - R
    
    N = len(nodes)
    L_col = np.full(N, L, dtype=np.float32)
    H_col = np.full(N, H, dtype=np.float32)
    R_col = np.full(N, R, dtype=np.float32)
    nu_col = np.full(N, nu, dtype=np.float32)
    
    X = np.column_stack([x, y, d, L_col, H_col, R_col, nu_col])
    Y = solution
    case_ids = np.full(N, case_id, dtype=np.int32)
    
    all_X.append(X)
    all_Y.append(Y)
    all_case_ids.append(case_ids)
    
    # NOVO: acumular nodes e seus case_ids (por nó)
    all_nodes_raw.append(nodes)                          # (N, 2)
    all_nodes_case_ids.append(case_ids)                  # (N,)

    # ------------ NOVO: ler e acumular tensões por elemento ------------
    # Espera-se que o JSON tenha: "sigma_elements": [[σ_xx, σ_yy, τ_xy], ...]
    if "sigma_elements" in data:
        sigma_elems = np.array(data["sigma_elements"], dtype=np.float32)  # (ne, 3)
        ne = sigma_elems.shape[0]
        sigma_case_ids = np.full(ne, case_id, dtype=np.int32)

        all_sigma.append(sigma_elems)
        all_sigma_case_ids.append(sigma_case_ids)
    # Se não tiver, simplesmente não adiciona nada (mantém compatibilidade)
    # -------------------------------------------------------------------

    # NOVO: acumular connectividade de elementos
    if "elements" in data:
        elems = np.array(data["elements"], dtype=np.int32)  # (ne_case, 3)
        ne_case = elems.shape[0]
        elem_case_ids = np.full(ne_case, case_id, dtype=np.int32)

        all_elements.append(elems)
        all_elements_case_ids.append(elem_case_ids)

# Concatenar (nó a nó)
X = np.vstack(all_X).astype(np.float32)
Y = np.vstack(all_Y).astype(np.float32)
case_indices = np.concatenate(all_case_ids).astype(np.int32)

# NOVO: concatenar nodes e seus case_ids
nodes_all = np.vstack(all_nodes_raw).astype(np.float32)              # (N_total, 2)
nodes_case_indices = np.concatenate(all_nodes_case_ids).astype(np.int32)

# Concatenar tensões (se existirem)
if all_sigma:
    Sigma = np.vstack(all_sigma).astype(np.float32)    # (N_elem_total, 3)
    
    S_max = np.max(np.abs(Sigma), axis = 0)
    Sigma_norm = Sigma / (S_max + 1e-8)
    
    sigma_case_indices = np.concatenate(all_sigma_case_ids).astype(np.int32)
else:
    Sigma = None
    sigma_case_indices = None

if all_elements:
    elements_all = np.vstack(all_elements).astype(np.int32)                 # (N_elem_total, 3)
    elements_case_indices = np.concatenate(all_elements_case_ids).astype(np.int32)
else:
    elements_all = None
    elements_case_indices = None

# Normalizar (com epsilon) — só nas features nodais, igual antes
X_max = np.max(np.abs(X[:, :6]), axis=0)
X[:, :6] = X[:, :6] / (X_max + 1e-8)

# Salvar
if Sigma is not None:
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        Y=Y,
        case_indices=case_indices,
        X_max=X_max,
        Sigma=Sigma_norm,
        sigma_case_indices=sigma_case_indices,
        S_max=S_max,
        # NOVO: salvar nodes e elements
        nodes=nodes_all,
        nodes_case_indices=nodes_case_indices,
        elements=elements_all,
        elements_case_indices=elements_case_indices
    )
else:
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        Y=Y,
        case_indices=case_indices,
        X_max=X_max,
        # NOVO: salvar nodes e elements mesmo sem Sigma
        nodes=nodes_all,
        nodes_case_indices=nodes_case_indices,
        elements=elements_all,
        elements_case_indices=elements_case_indices
    )

print(f"Salvo: {X.shape[0]} nós, {len(np.unique(case_indices))} casos")
if Sigma is not None:
    print(f"Salvo também {Sigma.shape[0]} elementos com tensões.")
print(f"Nodes salvos: {nodes_all.shape[0]} nós (nodes), {elements_all.shape[0] if elements_all is not None else 0} elementos")