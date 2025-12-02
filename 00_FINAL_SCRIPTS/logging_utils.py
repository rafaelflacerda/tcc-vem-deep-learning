# logging_utils.py

from pathlib import Path
from datetime import datetime
import torch

def setup_experiment_logging(
    project_root: Path,
    dataset_name: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    early_stopping: bool,
    patience: int
):
    """
    Cria pasta do experimento, arquivo de log e retorna:
      - experiment_dir (Path)
      - experiment_name (str)
      - log_print (função de logging)

    Já escreve o cabeçalho padrão no log.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{timestamp}"

    experiment_dir = project_root / "models" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    log_file = experiment_dir / "training_log.txt"

    def log_print(message: str, file=log_file):
        print(message)
        with open(file, 'a') as f:
            f.write(message + '\n')

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    header = f"""
{'='*60}
VEM Deep Learning Training
{'='*60}
Data/Hora Início: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {dataset_name}.npz
Experimento: {experiment_name}

HIPERPARÂMETROS:
  Arquitetura: 6 camadas [256-256-128-128-64-64]
  Batch size: {batch_size}
  Learning rate: {learning_rate}
  Optimizer: AdamW (weight_decay={weight_decay})
  Scheduler: CosineAnnealingLR
  Epochs: {epochs}
  Precision: FP16
  Device: {device_name}
  Early Stopping: {'Ativo (patience=' + str(patience) + ')' if early_stopping else 'Desativado'}
  
{'='*60}
"""
    log_print(header)

    # também já escreve cabeçalho da tabela
    log_print("\n" + "="*80)
    log_print(f"{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'LR':<12}{'Time(s)':<10}{'Best':<5}")
    log_print("="*80)

    return experiment_dir, experiment_name, log_print