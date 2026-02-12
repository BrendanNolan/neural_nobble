#!/usr/bin/env python3

import optuna
from pathlib import Path
import re
import subprocess
import time


def objective(trial):
    rng_seed = trial.suggest_int("rng_seed", 0, 1 << 32)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 16, 256, log=True)
        layers.append(layer_size)
    loss = run_mnist(rng_seed, layers)
    return loss


def run_mnist(rng_seed, layers):
    project_root = Path(__file__).resolve().parent.parent
    layers_str = ",".join(str(s) for s in layers)
    proc_result = subprocess.run(["cargo", "run", "--release", "--example", "mnist", "--",
        "--rng-seed", f"{rng_seed}", "--layers", layers_str],
        cwd=project_root,
        capture_output=True,
        text=True)
    if proc_result.returncode != 0:
        raise optuna.TrialPruned()
    output = proc_result.stdout
    hits = int(re.search(r"Hits: (\d+)", output).group(1))
    print()
    print(f"Hits: {hits}")
    return hits


if __name__ == "__main__":
    sample = optuna.create_study(direction="maximize")
    sample.optimize(objective, n_trials=500)
    print(sample.best_params)
    print(sample.best_value)

