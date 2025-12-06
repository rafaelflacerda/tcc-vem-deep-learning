import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig

# Importa o trainable do seu arquivo
from trainable import trainable

# Caminho raiz sempre seguro
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":

    ray.init()

    # -----------------------------
    # ASHA Scheduler (correto)
    # -----------------------------
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="val_w_mse",
        mode="min",
        max_t=200,
        grace_period=10,
        reduction_factor=3
    )

    # -----------------------------
    # Espaço de busca
    # -----------------------------
    search_space = {
        "LR_inicial": tune.loguniform(1e-5, 5e-3),
        "batch_size": tune.choice([128, 256, 512, 768]),
        "hidden_dim": tune.choice([256, 384, 448, 512]),
        "lambda_theta": tune.choice([1.0, 5.0]),
        "weight_decay": tune.loguniform(1e-7, 1e-4),
        "epochs": 200,
        "n_samples": 10000,
    }

    # -----------------------------
    # Tuner
    # -----------------------------
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=20,    # Ajustável
        ),
        run_config=RunConfig(
            storage_path=os.path.join(PROJECT_ROOT, "outputs", "ray_results"),
            name="asha_beam",
        )
    )

    # -----------------------------
    # Roda o ASHA
    # -----------------------------
    results = tuner.fit()

    best = results.get_best_result(metric="val_w_mse", mode="min")
    print("\n===== MELHOR RESULTADO =====")
    print("Config:", best.config)
    print("val_w_mse:", best.metrics["val_w_mse"])