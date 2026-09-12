"""Diagnose the two failing seeds of Round 2.

F3 run4 (seed 11004) stuck at 3.99; F9 run2 (seed 14002) stuck at 3079.
Logs per-generation improvement structure to see whether stagnation
(no-improvement streaks) ever reaches the re-seed threshold H=6.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA
from benchmarks.cec_suite import get_benchmark_suite

DIM = 10


class Logger:
    """Wrap the objective to count evaluations; algorithm state read after."""

    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)


def diagnose(prob_id, run_idx):
    suite = get_benchmark_suite(dim=DIM)
    func = suite[prob_id - 1]
    seed = 10000 + (prob_id - 1) * 500 + run_idx
    opt = RMA_EA(
        objective_func=func, dim=DIM,
        lower_bound=func.bounds[0], upper_bound=func.bounds[1],
        max_iter=1000, seed=seed,
    )

    # Monkey-patch internals via a traced optimize: simplest is to re-run the loop
    # with instrumentation by copying key statistics from history + a patched RNG.
    # Instead: run once and reconstruct barren streaks from history is impossible,
    # so we patch ParameterMemory.update_memory calls (fires iff improvement).
    events = []  # (iteration, global_best)

    orig_update = opt.memory.update_memory
    n_improved_gens = [0]
    state = {"gen": 0}

    def patched_update(**kwargs):
        n = len(kwargs.get("successful_F", []))
        n_improved_gens[0] += 1 if n > 0 else 0
        state["improved"] = n > 0
        return orig_update(**kwargs)

    # We cannot see per-gen detail this way; instead track global best history.
    res = opt.optimize()
    hist = res.history_fitness
    print(f"\n===== F{prob_id} run{run_idx} (seed {seed}) =====")
    print(f"final error: {max(0.0, res.best_f - func.bias):.4e}, iterations: {res.iterations}, fes: {res.fes}")

    # Reconstruct: generation-level global-best improvements
    improvements = [i for i in range(1, len(hist)) if hist[i] < hist[i - 1]]
    print(f"total best-improving generations: {len(improvements)} / {len(hist) - 1}")
    # longest streak without global-best improvement
    best_streak = cur = 0
    for i in range(1, len(hist)):
        if hist[i] >= hist[i - 1]:
            cur += 1
            best_streak = max(best_streak, cur)
        else:
            cur = 0
    print(f"longest streak WITHOUT global-best improvement: {best_streak} generations")
    # error at 25/50/75% of budget
    for frac in (0.25, 0.5, 0.75):
        k = int(frac * (len(hist) - 1))
        print(f"  at iter {k} ({int(frac*100)}%): best err {max(0.0, hist[k] - func.bias):.4e}")
    # when was the last improvement
    if improvements:
        print(f"last global-best improvement at iteration {improvements[-1]}")


if __name__ == "__main__":
    diagnose(3, 4)
    diagnose(9, 2)
