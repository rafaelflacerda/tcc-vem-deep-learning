# preprocess_dataset.py

import numpy as np
import torch
import os

def preprocess_npz(input_npz, output_npz, scalers_npz):
    print(f"[INFO] Lendo dataset bruto: {input_npz}")
    data = np.load(input_npz)

    X = data["X"].astype(np.float32)
    y = data["Y"].astype(np.float32)

    # 1) Calcula scalers
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0)
    y_mean = y.mean(axis=0)
    y_std  = y.std(axis=0)

    # 2) Normaliza
    X_norm = (X - X_mean) / (X_std + 1e-8)
    y_norm = (y - y_mean) / (y_std + 1e-8)

    # 3) Salva dataset já normalizado (pode ser float32 mesmo)
    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez(
        output_npz,
        X=X_norm.astype(np.float32),
        Y=y_norm.astype(np.float32),
    )
    print(f"[OK] Dataset pré-processado salvo em: {output_npz}")

    # 4) Salva scalers separados
    os.makedirs(os.path.dirname(scalers_npz), exist_ok=True)
    np.savez(
        scalers_npz,
        X_mean=X_mean.astype(np.float32),
        X_std=X_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )
    print(f"[OK] Scalers salvos em: {scalers_npz}")


if __name__ == "__main__":
    # Ajusta esses caminhos conforme o seu projeto
    PROJECT_ROOT = "/workspace/tcc-vem-deep-learning"  # ou usa os.path como no train
    N_SAMPLES = 1000000

    raw_npz = os.path.join(
        PROJECT_ROOT,
        "00_PROBLEMA_UNIDIMENSIONAL",
        "dataset",
        "npz",
        f"beam_dataset_{N_SAMPLES}_samples.npz",
    )

    preprocessed_npz = os.path.join(
        PROJECT_ROOT,
        "00_PROBLEMA_UNIDIMENSIONAL",
        "dataset",
        "npz_preprocessed",
        f"beam_dataset_{N_SAMPLES}_samples_preproc.npz",
    )

    scalers_npz = os.path.join(
        PROJECT_ROOT,
        "00_PROBLEMA_UNIDIMENSIONAL",
        "dataset",
        "npz_preprocessed",
        f"beam_scalers_{N_SAMPLES}_samples.npz",
    )

    preprocess_npz(raw_npz, preprocessed_npz, scalers_npz)
