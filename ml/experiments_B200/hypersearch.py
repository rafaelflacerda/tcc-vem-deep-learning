import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import json
import time
import datetime

import torch  # para detectar quantas GPUs existem

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

TRAIN_SCRIPT = "train-Copy1.py"  # caminho para o seu script de treino

# Valores base (herdados do caso com 2500 samples, adaptados para 10000)
BASE_N_SAMPLES = 10000    # só informativo (o train.py usa N_SAMPLES fixo)
BASE_EPOCHS = 200
BASE_HIDDEN_DIM = 384
BASE_DROPOUT_P = 0.10
BASE_LAMBDA_THETA = 5.0
BASE_WEIGHT_DECAY = 1e-4
BASE_BATCH_SIZE = 256
BASE_LR_INICIAL = 0.0014

# Quantos treinos em paralelo por etapa.
# Sugestão:
#   - cenários médios (10k samples, 200 epochs): MAX_WORKERS = 2 * N_GPUS
#   - cenários muito pesados: MAX_WORKERS = N_GPUS
# Aqui vamos configurar dinamicamente no main() com base em N_GPUS.
MAX_WORKERS = 1  # será sobrescrito no main()


# ============================================================
# ESTRUTURA DE PARÂMETROS E MÉTRICAS
# ============================================================

@dataclass
class HyperParams:
    n_samples: int     # só pra registro
    epochs: int
    hidden_dim: int
    dropout_p: float
    lambda_theta: float
    weight_decay: float
    batch_size: int
    lr_inicial: float
    device_id: Optional[int] = None  # GPU lógica a ser usada (0, 1, 2, ...)


@dataclass
class TrainMetrics:
    best_val_sobolev_loss: float
    best_val_w_mse: float
    epoch_of_best_val: int
    final_val_sobolev_loss: float
    final_val_w_mse: float
    tempo_medio_por_epoch: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainMetrics":
        return TrainMetrics(
            best_val_sobolev_loss=float(d["best_val_sobolev_loss"]),
            best_val_w_mse=float(d["best_val_w_mse"]),
            epoch_of_best_val=int(d["epoch_of_best_val"]),
            final_val_sobolev_loss=float(d["final_val_sobolev_loss"]),
            final_val_w_mse=float(d["final_val_w_mse"]),
            tempo_medio_por_epoch=float(d["tempo_medio_por_epoch"]),
        )


@dataclass
class ExperimentResult:
    params: HyperParams
    metrics: TrainMetrics
    stdout: str
    stderr: str
    elapsed_s: float


# Registro global de todos os experimentos
ALL_EXPERIMENTS: List[Dict[str, Any]] = []


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def assign_devices_round_robin(configs: List[HyperParams], n_gpus: int) -> None:
    """
    Atribui device_id às configs em esquema round-robin:
      cfg0 -> GPU 0
      cfg1 -> GPU 1
      cfg2 -> GPU 2
      ...
      cfgN -> GPU (N % n_gpus)
    """
    if n_gpus <= 0:
        raise RuntimeError("Nenhuma GPU CUDA encontrada.")

    for i, cfg in enumerate(configs):
        cfg.device_id = i % n_gpus


def build_command(params: HyperParams) -> List[str]:
    """
    Monta o comando para chamar o train.py.

    ATENÇÃO: usa exatamente os argumentos definidos no seu train.py.
    """
    cmd = [
        "python3", TRAIN_SCRIPT,
        "--LR_inicial", str(params.lr_inicial),
        "--batch_size", str(params.batch_size),
        "--lambda_theta", str(params.lambda_theta),
        "--hidden_dim", str(params.hidden_dim),
        "--dropout", str(params.dropout_p),
        "--weight_decay", str(params.weight_decay),
        "--epochs", str(params.epochs),
    ]
    return cmd


def parse_metrics_from_stdout(stdout: str) -> TrainMetrics:
    """
    Lê as métricas a partir das DUAS últimas linhas do stdout do train.py:

      linha -2: header CSV
      linha -1: valores CSV

    Exemplo:
      best_val_sobolev_loss;best_val_w_mse;epoch_of_best_val;final_val_sobolev_loss;final_val_w_mse;tempo_medio_por_epoch
      0.00123;4.56e-06;123;0.00180;5.00e-06;0.456
    """
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]

    if len(lines) < 2:
        raise ValueError("stdout não tem linhas suficientes para achar o CSV final de métricas.")

    header = lines[-2]
    values = lines[-1]

    expected_header = "best_val_sobolev_loss;best_val_w_mse;epoch_of_best_val;final_val_sobolev_loss;final_val_w_mse;tempo_medio_por_epoch"
    if header != expected_header:
        raise ValueError(
            "Header CSV inesperado nas métricas.\n"
            f"Esperado: {expected_header}\n"
            f"Encontrado: {header}"
        )

    parts = values.split(";")
    if len(parts) != 6:
        raise ValueError(
            f"Linha de valores CSV não tem 6 colunas: {values}"
        )

    keys = [
        "best_val_sobolev_loss",
        "best_val_w_mse",
        "epoch_of_best_val",
        "final_val_sobolev_loss",
        "final_val_w_mse",
        "tempo_medio_por_epoch",
    ]
    data: Dict[str, Any] = {}
    for key, val in zip(keys, parts):
        data[key] = val

    return TrainMetrics.from_dict(data)


def run_single_training(params: HyperParams) -> ExperimentResult:
    cmd = build_command(params)
    print(f"\n[RUN] (GPU {params.device_id}) {cmd}")

    # Clona o ambiente e força qual GPU o processo vai ver
    env = os.environ.copy()
    if params.device_id is not None:
        # IMPORTANTE: cada processo verá SOMENTE essa GPU como "cuda:0"
        env["CUDA_VISIBLE_DEVICES"] = str(params.device_id)

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"[ERRO] Comando falhou (exit code {proc.returncode}).")
        print(proc.stderr)
        raise RuntimeError(f"Treino falhou para params={params}")

    metrics = parse_metrics_from_stdout(proc.stdout)
    print(
        f"[OK] (GPU {params.device_id}) "
        f"batch={params.batch_size} lr={params.lr_inicial:.6g} "
        f"lambda={params.lambda_theta} hidden={params.hidden_dim} "
        f"dropout={params.dropout_p} wd={params.weight_decay:.1e} | "
        f"best_val_w_mse={metrics.best_val_w_mse:.4e} "
        f"best_val_sobolev_loss={metrics.best_val_sobolev_loss:.4e}"
    )

    return ExperimentResult(
        params=params,
        metrics=metrics,
        stdout=proc.stdout,
        stderr=proc.stderr,
        elapsed_s=elapsed,
    )


def run_experiments_parallel(configs: List[HyperParams], max_workers: int) -> List[ExperimentResult]:
    """
    Roda vários treinos em paralelo (até max_workers simultaneamente).
    """
    results: List[ExperimentResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {executor.submit(run_single_training, cfg): cfg for cfg in configs}
        for future in as_completed(future_to_cfg):
            cfg = future_to_cfg[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"[FALHA] Treino com params={cfg} falhou: {e}")
    return results


def pick_best_config(results: List[ExperimentResult]) -> ExperimentResult:
    """
    Critério:
      1) menor best_val_w_mse
      2) empate: menor best_val_sobolev_loss
    """
    if not results:
        raise ValueError("Nenhum resultado disponível para escolher o melhor.")

    best: Optional[ExperimentResult] = None
    for r in results:
        if best is None:
            best = r
        else:
            if r.metrics.best_val_w_mse < best.metrics.best_val_w_mse:
                best = r
            elif (
                math.isclose(r.metrics.best_val_w_mse, best.metrics.best_val_w_mse, rel_tol=1e-8, abs_tol=0.0)
                and r.metrics.best_val_sobolev_loss < best.metrics.best_val_sobolev_loss
            ):
                best = r
    assert best is not None
    return best


def print_stage_summary(stage_name: str, results: List[ExperimentResult], best: ExperimentResult):
    print("\n" + "=" * 70)
    print(f"Resumo {stage_name}")
    print("=" * 70)
    for r in results:
        p = r.params
        m = r.metrics
        print(
            f"(GPU {p.device_id}) "
            f"batch={p.batch_size:4d} lr={p.lr_inicial:.6g} "
            f"lambda={p.lambda_theta:4.2f} hidden={p.hidden_dim:3d} "
            f"dropout={p.dropout_p:.4f} wd={p.weight_decay:.1e} | "
            f"best_w_mse={m.best_val_w_mse:.4e} "
            f"best_sobolev={m.best_val_sobolev_loss:.4e} "
            f"epoch_best={m.epoch_of_best_val} "
            f"t_epoch={m.tempo_medio_por_epoch:.3f}s"
        )

    print("-" * 70)
    print("Melhor configuração da etapa:")
    print(json.dumps({
        "params": asdict(best.params),
        "metrics": asdict(best.metrics),
    }, indent=2))
    print("=" * 70 + "\n")


def log_stage_results(stage_name: str, results: List[ExperimentResult]):
    """
    Salva todos os resultados de uma etapa na lista global ALL_EXPERIMENTS.
    """
    for r in results:
        entry = {
            "stage": stage_name,
            "params": asdict(r.params),
            "metrics": asdict(r.metrics),
            "elapsed_s": r.elapsed_s,
        }
        ALL_EXPERIMENTS.append(entry)


# ============================================================
# ETAPAS DA OTIMIZAÇÃO
# ============================================================

def etapa_1_batch_lr(base: HyperParams, n_gpus: int) -> Tuple[HyperParams, List[ExperimentResult]]:
    """
    Etapa 1:
      - varia batch_size = 256, 384, 512, 640, 768
      - ajusta lr_inicial: lr_new = lr_base * sqrt(batch_new / 256)
      - escolhe BATCH_SIZE_AJUSTADO e LR_AJUSTADA
    """
    print("\n===== ETAPA 1: varredura em batch_size + lr (regra da raiz) =====")

    batch_values = [256, 384, 512, 640, 768]
    configs: List[HyperParams] = []
    for b in batch_values:
        factor = math.sqrt(b / BASE_BATCH_SIZE)
        lr_new = base.lr_inicial * factor
        cfg = HyperParams(
            n_samples=base.n_samples,
            epochs=base.epochs,
            hidden_dim=base.hidden_dim,
            dropout_p=base.dropout_p,
            lambda_theta=base.lambda_theta,
            weight_decay=base.weight_decay,
            batch_size=b,
            lr_inicial=lr_new,
        )
        configs.append(cfg)

    # Distribui as configs entre as GPUs
    assign_devices_round_robin(configs, n_gpus)

    results = run_experiments_parallel(configs, MAX_WORKERS)
    best = pick_best_config(results)

    stage_name = "ETAPA 1 (batch_size + lr)"
    print_stage_summary(stage_name, results, best)
    log_stage_results(stage_name, results)

    return best.params, results


def etapa_2_refina_lr(base: HyperParams, n_gpus: int) -> Tuple[HyperParams, List[ExperimentResult]]:
    """
    Etapa 2:
      - mantém batch_size ajustado
      - usa múltiplos da LR_AJUSTADA: 0.7225, 0.85, 1.0, 1.15, 1.3225
      - escolhe LR_NOVA
    """
    print("\n===== ETAPA 2: refino da learning rate em torno da LR_AJUSTADA =====")

    multipliers = [0.7225, 0.85, 1.0, 1.15, 1.3225]
    configs: List[HyperParams] = []
    for alpha in multipliers:
        lr_new = base.lr_inicial * alpha
        cfg = HyperParams(
            n_samples=base.n_samples,
            epochs=base.epochs,
            hidden_dim=base.hidden_dim,
            dropout_p=base.dropout_p,
            lambda_theta=base.lambda_theta,
            weight_decay=base.weight_decay,
            batch_size=base.batch_size,
            lr_inicial=lr_new,
        )
        configs.append(cfg)

    assign_devices_round_robin(configs, n_gpus)

    results = run_experiments_parallel(configs, MAX_WORKERS)
    best = pick_best_config(results)

    stage_name = "ETAPA 2 (refino LR)"
    print_stage_summary(stage_name, results, best)
    log_stage_results(stage_name, results)

    return best.params, results


def etapa_3_lambda_theta(base: HyperParams, n_gpus: int) -> Tuple[HyperParams, List[ExperimentResult]]:
    """
    Etapa 3:
      - mantém batch_size e lr_nova
      - varre lambda_theta = 2,3,4,5,6,7,8
      - escolhe LAMBDA_NOVO
    """
    print("\n===== ETAPA 3: varredura em lambda_theta =====")

    lambdas = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    configs: List[HyperParams] = []
    for lam in lambdas:
        cfg = HyperParams(
            n_samples=base.n_samples,
            epochs=base.epochs,
            hidden_dim=base.hidden_dim,
            dropout_p=base.dropout_p,
            lambda_theta=lam,
            weight_decay=base.weight_decay,
            batch_size=base.batch_size,
            lr_inicial=base.lr_inicial,
        )
        configs.append(cfg)

    assign_devices_round_robin(configs, n_gpus)

    results = run_experiments_parallel(configs, MAX_WORKERS)
    best = pick_best_config(results)

    stage_name = "ETAPA 3 (lambda_theta)"
    print_stage_summary(stage_name, results, best)
    log_stage_results(stage_name, results)

    return best.params, results


def etapa_4_hidden_dim(base: HyperParams, n_gpus: int) -> Tuple[HyperParams, List[ExperimentResult]]:
    """
    Etapa 4:
      - mantém batch_size, lr_nova, lambda_novo
      - varre hidden_dim = 256, 320, 384, 448, 512
      - escolhe HIDDEN_DIM_NOVO
    """
    print("\n===== ETAPA 4: varredura em hidden_dim =====")

    hidden_values = [256, 320, 384, 448, 512]
    configs: List[HyperParams] = []
    for h in hidden_values:
        cfg = HyperParams(
            n_samples=base.n_samples,
            epochs=base.epochs,
            hidden_dim=h,
            dropout_p=base.dropout_p,
            lambda_theta=base.lambda_theta,
            weight_decay=base.weight_decay,
            batch_size=base.batch_size,
            lr_inicial=base.lr_inicial,
        )
        configs.append(cfg)

    assign_devices_round_robin(configs, n_gpus)

    results = run_experiments_parallel(configs, MAX_WORKERS)
    best = pick_best_config(results)

    stage_name = "ETAPA 4 (hidden_dim)"
    print_stage_summary(stage_name, results, best)
    log_stage_results(stage_name, results)

    return best.params, results


def etapa_5_dropout(base: HyperParams, n_gpus: int) -> Tuple[HyperParams, List[ExperimentResult]]:
    """
    Etapa 5:
      - mantém batch_size, lr_nova, lambda_novo, hidden_dim_novo
      - varre dropout_p = 0.075, 0.0875, 0.10, 0.1125, 0.125
      - escolhe DROPOUT_NOVO
    """
    print("\n===== ETAPA 5: varredura em dropout_p =====")

    dropouts = [0.0750, 0.0875, 0.1000, 0.1125, 0.1250]
    configs: List[HyperParams] = []
    for d in dropouts:
        cfg = HyperParams(
            n_samples=base.n_samples,
            epochs=base.epochs,
            hidden_dim=base.hidden_dim,
            dropout_p=d,
            lambda_theta=base.lambda_theta,
            weight_decay=base.weight_decay,
            batch_size=base.batch_size,
            lr_inicial=base.lr_inicial,
        )
        configs.append(cfg)

    assign_devices_round_robin(configs, n_gpus)

    results = run_experiments_parallel(configs, MAX_WORKERS)
    best = pick_best_config(results)

    stage_name = "ETAPA 5 (dropout_p)"
    print_stage_summary(stage_name, results, best)
    log_stage_results(stage_name, results)

    return best.params, results


def etapa_6_weight_decay(base: HyperParams, n_gpus: int) -> Tuple[HyperParams, List[ExperimentResult]]:
    """
    Etapa 6:
      - mantém batch_size, lr_nova, lambda_novo, hidden_dim_novo, dropout_novo
      - varre weight_decay = [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
      - escolhe WEIGHT_DECAY_NOVO
    """
    print("\n===== ETAPA 6: varredura em weight_decay =====")

    wds = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
    configs: List[HyperParams] = []
    for wd in wds:
        cfg = HyperParams(
            n_samples=base.n_samples,
            epochs=base.epochs,
            hidden_dim=base.hidden_dim,
            dropout_p=base.dropout_p,
            lambda_theta=base.lambda_theta,
            weight_decay=wd,
            batch_size=base.batch_size,
            lr_inicial=base.lr_inicial,
        )
        configs.append(cfg)

    assign_devices_round_robin(configs, n_gpus)

    results = run_experiments_parallel(configs, MAX_WORKERS)
    best = pick_best_config(results)

    stage_name = "ETAPA 6 (weight_decay)"
    print_stage_summary(stage_name, results, best)
    log_stage_results(stage_name, results)

    return best.params, results


# ============================================================
# MAIN
# ============================================================

def main():
    global MAX_WORKERS

    # Detecta quantas GPUs existem
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("Nenhuma GPU CUDA detectada.")
    print(f"Detectadas {n_gpus} GPUs CUDA para o hypersearch.")

    # Aqui você escolhe a política de MAX_WORKERS.
    # Para o caso atual (10k samples, 200 epochs), pode ser mais agressivo:
    MAX_WORKERS = 2 * n_gpus
    print(f"MAX_WORKERS definido como {MAX_WORKERS}.")

    # Parâmetros iniciais (aqueles definidos por você para 2500 samples,
    # agora usando epochs=200 e N_SAMPLES=10000)
    base_params = HyperParams(
        n_samples=BASE_N_SAMPLES,
        epochs=BASE_EPOCHS,
        hidden_dim=BASE_HIDDEN_DIM,
        dropout_p=BASE_DROPOUT_P,
        lambda_theta=BASE_LAMBDA_THETA,
        weight_decay=BASE_WEIGHT_DECAY,
        batch_size=BASE_BATCH_SIZE,
        lr_inicial=BASE_LR_INICIAL,
    )

    print("Parâmetros iniciais:")
    print(json.dumps(asdict(base_params), indent=2))

    # Etapa 1: batch_size + lr pela regra da raiz
    params_e1, _ = etapa_1_batch_lr(base_params, n_gpus)

    # Etapa 2: refino da LR
    params_e2, _ = etapa_2_refina_lr(params_e1, n_gpus)

    # Etapa 3: lambda_theta
    params_e3, _ = etapa_3_lambda_theta(params_e2, n_gpus)

    # Etapa 4: hidden_dim
    params_e4, _ = etapa_4_hidden_dim(params_e3, n_gpus)

    # Etapa 5: dropout_p
    params_e5, _ = etapa_5_dropout(params_e4, n_gpus)

    # Etapa 6: weight_decay
    params_e6, _ = etapa_6_weight_decay(params_e5, n_gpus)

    print("\n\n================ PARÂMETROS FINAIS SELECIONADOS ================")
    print(json.dumps(asdict(params_e6), indent=2))
    print("================================================================\n")

    # Salvar todas as combinações testadas em um JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Diretório fixo para salvar resultados
    SAVE_DIR = "/workspace/treinamentos/hypersearch"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    filename = os.path.join(SAVE_DIR, f"hypersearch_results_{timestamp}.json")
    
    with open(filename, "w") as f:
        json.dump(ALL_EXPERIMENTS, f, indent=2)
    
    print(f"✅ Resultados de todas as combinações salvos em: {filename}")


if __name__ == "__main__":
    main()
