import numpy as np
import json

# Nome do arquivo NPZ na mesma pasta
NPZ_FILE = "arquivo.npz"
JSON_FILE = "arquivo_convertido.json"

def npz_to_json(npz_path, json_path):
    # Carrega o npz
    data = np.load(npz_path)

    # Dicionário para armazenar tudo
    json_dict = {}

    # Para cada array no npz
    for key in data.files:
        array = data[key]

        # Converte numpy array para lista Python
        json_dict[key] = array.tolist()

    # Salva como JSON
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)

    print(f"Conversão concluída! JSON salvo em: {json_path}")


if __name__ == "__main__":
    npz_to_json(NPZ_FILE, JSON_FILE)