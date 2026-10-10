"""Ray Tune grid/random search on a synthetic loss, logged to MLflow.

Requires a running MLflow server: mlflow server --host 127.0.0.1 --port 8080
"""

import ray
import torch
import torch.nn.functional as F
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback

TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "ray-tune-experiments"


def loss_fn(step, width, height, activation):
    pre_act = torch.tensor((0.1 + width * step / 100) ** (-1) + height * 0.1)
    if "relu" == activation:
        return F.relu(pre_act)
    elif "tanh" == activation:
        return F.tanh(pre_act)


def trainable(config):
    width, height, activation = config["width"], config["height"], config["activation"]

    for step in range(config["steps"]):
        loss = loss_fn(step, width, height, activation)
        tune.report({"iterations": step, "mean_loss": loss.item()})


def main():
    # Default would start one worker per core and run into Ray's OOM killer on this machine
    ray.init(num_cpus=4)

    search_space = {
        "steps": 100,
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
        "activation": tune.grid_search(["relu", "tanh"]),
    }

    tune_config = tune.TuneConfig(
        metric="mean_loss",
        mode="min",
        search_alg=tune.search.basic_variant.BasicVariantGenerator(),
        scheduler=None,
        num_samples=20,
    )

    run_config = tune.RunConfig(
        name="MLflow logging experiment",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=TRACKING_URI,
                experiment_name=EXPERIMENT_NAME,
                save_artifact=True,
            )
        ],
    )

    tuner = tune.Tuner(
        trainable=trainable,
        tune_config=tune_config,
        run_config=run_config,
        param_space=search_space,
    )
    results = tuner.fit()

    best = results.get_best_result()
    print(f"Best mean_loss: {best.metrics['mean_loss']:.4f}")
    print(f"Best config:    {best.config}")


if __name__ == "__main__":
    main()
