import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

N_SAMPLES = 2500

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

JSON_DIR = PROJECT_ROOT / "00_PROBLEMA_UNIDIMENSIONAL" / "dataset" / f"beam_results_{N_SAMPLES}_samples"
OUTPUT_FILE = PROJECT_ROOT / "00_PROBLEMA_UNIDIMENSIONAL" / "dataset" / "npz" / f"beam_dataset_{N_SAMPLES}_samples.npz"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

json_files = sorted(JSON_DIR.glob("*.json"))

all_X = []
all_Y = []

for jfile in tqdm(json_files, desc="Processando JSONs"):
    with open(jfile) as f:
        data = json.load(f)
    
    E = data['E']
    I = data['I']
    q = data['q']
    L = data['L']
    
    positions = np.array(data['positions'], dtype=np.float32)
    w = np.array(data['w'], dtype=np.float32)
    theta = np.array(data['theta'], dtype=np.float32)
    
    # Transformação log (melhora aprendizado)
    E_log = np.log10(E)
    I_log = np.log10(I)
    
    N = len(positions)
    
    # Features: [x, log10(E), log10(I), q, L]
    X = np.column_stack([
        positions,
        np.full(N, E_log, dtype=np.float32),
        np.full(N, I_log, dtype=np.float32),
        np.full(N, q, dtype=np.float32),
        np.full(N, L, dtype=np.float32)
    ])
    
    # Targets: [w, theta]
    Y = np.column_stack([w, theta])
    
    all_X.append(X)
    all_Y.append(Y)

# Concatenar (sem normalização!)
X = np.vstack(all_X).astype(np.float32)
Y = np.vstack(all_Y).astype(np.float32)

# Salvar raw data
np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    Y=Y
)

print(f"{X.shape[0]} samples, {len(json_files)} cases")