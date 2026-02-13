#!/usr/bin/env python3

import optuna
from pathlib import Path
import re
import subprocess
import time


def objective(trial):
    rng_seed = trial.suggest_int("rng_seed", 0, 1 << 32)
    n_layers = trial.suggest_int("n_layers", 10000, 10000)
    layers = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 10000, 10000, log=True)
        layers.append(layer_size)
    loss = run_mnist(rng_seed, layers)
    return loss


def run_mnist(rng_seed, layers):
    project_root = Path(__file__).resolve().parent.parent
    layers_str = ",".join(str(s) for s in layers)
    start_time = time.time()
    proc_result = subprocess.run(["cargo", "run", "--features", "neural_nobble_log", "--release", "--example", "mnist", "--",
        "--rng-seed", f"{rng_seed}", "--layers", layers_str],
        cwd=project_root,
        capture_output=True,
        text=True)
    total_time = time.time() - start_time
    if proc_result.returncode != 0:
        raise optuna.TrialPruned()
    output = proc_result.stdout
    hits = int(re.search(r"Hits: (\d+)", output).group(1))
    misses = int(re.search(r"Misses: (\d+)", output).group(1))
    if misses == 0:
        misses = 1 / (1 << 32)
    miss_rate = misses / (hits + misses)
    print(flush=True)
    print(f"Time: {total_time}s, Hits: {hits}, Misses: {misses}", flush=True)
    baseline_time = 3
    return (total_time / baseline_time) * miss_rate


if __name__ == "__main__":
    sample = optuna.create_study(direction="minimize")
    sample.optimize(objective, n_trials=500)
    print(sample.best_params, flush=True)
    print(sample.best_value, flush=True)

