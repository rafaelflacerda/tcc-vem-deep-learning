import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

N_SAMPLES = 100
NPZ_FILE = os.path.join(PROJECT_ROOT, "00_PROBLEMA_UNIDIMENSIONAL", "dataset", "npz", f"beam_dataset_{N_SAMPLES}_samples.npz")

data = np.load(NPZ_FILE)
print(data.files)
