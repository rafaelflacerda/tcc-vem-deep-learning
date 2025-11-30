import numpy as np
import json
from pathlib import Path

# Configuração
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

NPZ_FILE = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_npz/meshes_500_samples.npz"
JSON_DIR = PROJECT_ROOT / "00_URGENTE/malha/training_dataset_json/meshes_500_samples"

# Carregar NPZ
print("Carregando NPZ...")
data = np.load(NPZ_FILE)

# 1. Verificar chaves
print("\n=== Chaves ===")
print(f"Chaves encontradas: {list(data.keys())}")
assert 'X' in data, "Falta X"
assert 'Y' in data, "Falta Y"
assert 'case_indices' in data, "Falta case_indices"
assert 'X_max' in data, "Falta X_max"
print("✓ Todas as chaves presentes")

# 2. Verificar shapes
print("\n=== Shapes ===")
X = data['X']
Y = data['Y']
case_indices = data['case_indices']
X_max = data['X_max']

print(f"X: {X.shape}")
print(f"Y: {Y.shape}")
print(f"case_indices: {case_indices.shape}")
print(f"X_max: {X_max.shape}")

assert X.shape[1] == 7, "X deve ter 7 features"
assert Y.shape[1] == 2, "Y deve ter 2 targets"
assert X.shape[0] == Y.shape[0], "X e Y devem ter mesmo número de linhas"
assert X.shape[0] == len(case_indices), "case_indices deve ter mesmo tamanho"
print("✓ Shapes corretos")

# 3. Verificar valores
print("\n=== Valores ===")
assert not np.isnan(X).any(), "X tem NaN"
assert not np.isnan(Y).any(), "Y tem NaN"
assert not np.isinf(X).any(), "X tem Inf"
assert not np.isinf(Y).any(), "Y tem Inf"
print("✓ Sem NaN ou Inf")

# 4. Verificar case_indices
print("\n=== Cases ===")
unique_cases = np.unique(case_indices)
print(f"Casos únicos: {len(unique_cases)}")
print(f"Range: {unique_cases.min()} a {unique_cases.max()}")

# 5. Comparar com JSON original (primeiro caso)
print("\n=== Validação vs JSON ===")
first_json = sorted(JSON_DIR.glob("*.json"))[0]
with open(first_json) as f:
    original = json.load(f)

case_id = original['case_id']
mask = case_indices == case_id
X_case = X[mask]
Y_case = Y[mask]

# Reconstruir dados originais
nodes_original = np.array(original['nodes'])
solution_original = np.array(original['solution_u']).reshape(-1, 2)

print(f"JSON: {len(nodes_original)} nós")
print(f"NPZ:  {len(X_case)} nós")
assert len(X_case) == len(nodes_original), "Número de nós diferente"

# Verificar se u,v batem (desnormalizado)
Y_case_denorm = Y_case  # Já está desnormalizado (Y não é normalizado)
diff = np.abs(Y_case_denorm - solution_original).max()
print(f"Diferença máxima u,v: {diff:.2e}")
assert diff < 1e-5, f"Diferença muito grande: {diff}"

print("\n✓ NPZ validado com sucesso!")