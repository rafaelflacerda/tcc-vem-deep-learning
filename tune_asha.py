import os

# ====== REMOVER WARNINGS ANTES DO RAY INICIALIZAR ======
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"
os.environ["RAY_EXPERIMENTAL_NO_LOG_EXPORT"] = "1"

# Silenciar DeprecationWarning, FutureWarning e UserWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import RunConfig, CLIReporter
import ray_silence_warnings

# 👉 TensorBoard callback
from ray.tune.logger import TBXLoggerCallback

from trainable import trainable

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":

    ray.init(log_to_driver=False)

    reporter = CLIReporter(
        metric_columns=[
            "val_w_mse",
            "val_loss",
            "train_loss",
            "lr",
            "time_total_s"
        ],
        parameter_columns=[
            "LR_inicial",
            "batch_size",
            "hidden_dim",
            "weight_decay"
        ],
        sort_by_metric=True,
        metric="val_w_mse",
        max_progress_rows=40
    )

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="val_w_mse",
        mode="min",
        max_t=300,
        grace_period=12,
        reduction_factor=2
    )

    search_space = {
        "LR_inicial": tune.loguniform(4e-4, 1e-3),
        "batch_size": tune.choice([384, 416, 448, 480, 512, 544, 576]),
        "hidden_dim": tune.choice([384, 392, 400, 408, 416, 424, 432, 440, 448]),
        "lambda_theta": 1,
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "epochs": 300,
        "n_samples": 10000,
    }

    trainable_gpu = tune.with_resources(trainable, {"gpu": 1})

    tuner = tune.Tuner(
        trainable_gpu,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=300,
            max_concurrent_trials=5,
        ),
        run_config=RunConfig(
            storage_path=os.path.join(PROJECT_ROOT, "outputs", "ray_results"),
            name="asha_beam",

            # 👉 mantém seus logs organizados
            log_to_file=("stdout.log", "stderr.log"),

            # 👉 mantém CLIReporter
            progress_reporter=reporter,

            # 👉 ADICIONA TENSORBOARD
            callbacks=[
                TBXLoggerCallback()  
            ]
        )
    )

    results = tuner.fit()

    best = results.get_best_result(metric="val_w_mse", mode="min")
    print("\n===== MELHOR RESULTADO =====")
    print("Config:", best.config)
    print("val_w_mse:", best.metrics["val_w_mse"])