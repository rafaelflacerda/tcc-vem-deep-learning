import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Lista de tamanhos de dataset que você quer processar
SAMPLES_LIST = [10, 50, 100, 500, 1000, 2500]

# Caminho base (ajuste se necessário)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR  # ou SCRIPT_DIR.parent, dependendo de onde salvar o script

for SAMPLES in SAMPLES_LIST:
    print("\n" + "="*60)
    print(f"Processando SAMPLES = {SAMPLES}")
    print("="*60)

    JSON_DIR = PROJECT_ROOT / "00_URGENTE" / "malha" / "training_dataset_json_sigma" / f"meshes_{SAMPLES}_samples"

    if not JSON_DIR.exists():
        print(f"[AVISO] Pasta não encontrada: {JSON_DIR}")
        continue

    json_files = sorted(JSON_DIR.glob("result_data_*.json"))

    if len(json_files) == 0:
        print(f"[AVISO] Nenhum result_data_*.json encontrado em {JSON_DIR}")
        continue

    print(f"Encontrados {len(json_files)} arquivos em {JSON_DIR}")

    for jfile in tqdm(json_files, desc=f"SAMPLES {SAMPLES}"):
        with open(jfile) as f:
            data = json.load(f)
        
        # 1) Carregar nodes e elements
        nodes = np.array(data["nodes"], dtype=float)          # (N, 2)
        elements = np.array(data["elements"], dtype=int)      # (ne, 3)

        # 2) Ajuste de indexação (se estiver 1-based)
        if elements.min() == 1 and elements.max() == len(nodes):
            elements = elements - 1  # converte para 0-based

        # 3) Coordenadas dos nós de cada elemento: (ne, 3, 2)
        elem_nodes_coords = nodes[elements]  # indexing avançado

        # 4) Centróide de cada elemento (média das 3 linhas)
        centroids = elem_nodes_coords.mean(axis=1)  # (ne, 2)

        # 5) Salvar no JSON
        data["sigma_points"] = centroids.tolist()

        # 6) Regravar o arquivo
        with open(jfile, "w") as f:
            json.dump(data, f, indent=2)

    print(f"✓ Finalizado para SAMPLES = {SAMPLES}")