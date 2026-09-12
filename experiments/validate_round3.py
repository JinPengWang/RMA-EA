"""Round-3 Validation: Operator-Decoupled Chart Posteriors (Design Q2).

Compares RMA-EA v3.3 (crossover chart by count credit, diffusion chart by
improvement-mass credit) against the stored v3.2 full-benchmark results
(identical seed schedule), on F3 (Q's side-effect function) and the multimodal
weak problems, at both D10 and D30.

Protocol: seed = 10000 + (prob_id-1)*500 + run_idx; max_iter 1000 (D10) / 1500 (D30).
"""

import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA
from benchmarks.cec_suite import get_benchmark_suite

PROB_IDS = [3, 4, 5, 9]
N_RUNS = 10


def _worker(args):
    dim, max_iter, prob_id, run_idx = args
    suite = get_benchmark_suite(dim=dim)
    func = suite[prob_id - 1]
    seed = 10000 + (prob_id - 1) * 500 + run_idx
    opt = RMA_EA(
        objective_func=func, dim=dim,
        lower_bound=func.bounds[0], upper_bound=func.bounds[1],
        max_iter=max_iter, seed=seed,
    )
    res = opt.optimize()
    err = max(0.0, float(res.best_f - func.bias))
    return dim, prob_id, run_idx, 0.0 if err < 1e-8 else err


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w
    d = np.array(a) - np.array(b)
    if np.all(d == 0):
        return 1.0
    try:
        return float(w(a, b).pvalue)
    except ValueError:
        return 1.0


def main():
    results = {}
    tasks = [(dim, 1000 if dim == 10 else 1500, p, r)
             for dim in (10, 30) for p in PROB_IDS for r in range(N_RUNS)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            dim, prob_id, run_idx, err = f.result()
            results.setdefault((dim, prob_id), []).append(err)
            print(f"[{i + 1}/{len(tasks)}] D{dim} F{prob_id} run{run_idx}: {err:.4e}")

    refs = {}
    for dim in (10, 30):
        with open(os.path.join(os.path.dirname(__file__), "results", f"cec_results_D{dim}_v3.2_backup.json")) as fh:
            refs[dim] = json.load(fh)

    print("\n" + "=" * 96)
    print(f"{'Case':9s} {'v3.2 mean':>12s} {'v3.3 mean':>12s} {'p(v3.3~v3.2)':>13s} "
          f"{'v3.2 med':>10s} {'v3.3 med':>10s} {'L-SHADE mean':>12s}")
    print("-" * 96)
    for dim in (10, 30):
        for p in PROB_IDS:
            name = refs[dim]["problems"][p - 1]
            old = refs[dim]["all_errors"]["RMA-EA"][name]
            new = results[(dim, p)]
            ls = refs[dim]["all_errors"]["L-SHADE"][name]
            med = lambda e: sorted(e)[len(e) // 2]
            pv = wilcoxon(new, old)
            print(f"D{dim} F{p:<3d} {np.mean(old):>12.4f} {np.mean(new):>12.4f} {pv:>13.4f} "
                  f"{med(old):>10.4f} {med(new):>10.4f} {np.mean(ls):>12.4f}")
    print("=" * 96)

    out = {
        "prob_ids": PROB_IDS, "n_runs": N_RUNS, "variant": "Operator-decoupled posteriors (Design Q2)",
        "errors": {f"D{dim}_F{p}": results[(dim, p)] for dim in (10, 30) for p in PROB_IDS},
    }
    path = os.path.join(os.path.dirname(__file__), "results", "round3_designQ2_validation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
