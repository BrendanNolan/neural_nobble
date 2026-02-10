#!/usr/bin/env python3

import optuna
import re
import subprocess
import time


def objective(trial):
    rng_seed = trial.suggest_int("rng_seed", 0, 100)
    loss = run_mnist(rng_seed)
    return loss


def run_mnist(rng_seed):
    start = time.time()
    proc_result = subprocess.run(["cargo", "run", "--example", "mnist", "--", f"{rng_seed}"],
        capture_output=True,
        text=True)
    elapsed_seconds = time.time() - start
    if proc_result.returncode != 0:
        return pow(2, 999999999)
    output = proc_result.stdout
    hits = int(re.search(r"Hits: (\d+)", output).group(1))
    misses = int(re.search(r"Misses: (\d+)", output).group(1))
    print(f"Hits: {hits}, Misses: {misses}, Time: {elapsed_seconds} seconds")
    miss_rate = misses / (hits + misses)
    if miss_rate == 0:
        miss_rate = 0.0000001
    real_to_expected_seconds = elapsed_seconds / 3
    return real_to_expected_seconds * miss_rate


if __name__ == "__main__":
    sample = optuna.create_study(direction="minimize")
    sample.optimize(objective, n_trials=100)
    print(sample.best_params)
    print(sample.best_value)

