"""Round-2 Validation: Geodesic Re-seeding (GR) A/B Experiment.

Compares RMA-EA v3.2 (GDM + Geodesic Re-seeding) against the stored Round-1
results (GDM only, identical seed schedule) and the L-SHADE reference, on the
weak multimodal problems plus unimodal sanity checks.

Protocol: dim=10, max_iter=1000, seed = 10000 + (prob_id-1)*500 + run_idx.
"""

import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA
from benchmarks.cec_suite import get_benchmark_suite

PROB_IDS = [1, 3, 4, 5, 9, 10]
N_RUNS = 10
MAX_ITER = 1000
DIM = 10


def _worker(args):
    prob_id, run_idx = args
    suite = get_benchmark_suite(dim=DIM)
    func = suite[prob_id - 1]
    seed = 10000 + (prob_id - 1) * 500 + run_idx
    opt = RMA_EA(
        objective_func=func,
        dim=DIM,
        lower_bound=func.bounds[0],
        upper_bound=func.bounds[1],
        max_iter=MAX_ITER,
        geodesic_diffusion=True,
        geodesic_reseed=True,
        seed=seed,
    )
    res = opt.optimize()
    err = max(0.0, float(res.best_f - func.bias))
    return prob_id, run_idx, 0.0 if err < 1e-8 else err


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
    with open(os.path.join(os.path.dirname(__file__), "results", "round1_gdm_validation.json")) as fh:
        r1 = json.load(fh)
    with open(os.path.join(os.path.dirname(__file__), "results", "cec_results_D10.json")) as fh:
        stored = json.load(fh)

    tasks = [(p, r) for p in PROB_IDS for r in range(N_RUNS)]
    results = {p: [] for p in PROB_IDS}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            prob_id, run_idx, err = f.result()
            results[prob_id].append(err)
            print(f"[{i + 1}/{len(tasks)}] F{prob_id} run{run_idx}: {err:.4e}")

    print("\n" + "=" * 92)
    print(f"{'Prob':5s} {'R1(GDM) mean':>13s} {'R2(+GR) mean':>13s} {'p(R2~R1)':>9s} "
          f"{'R1 med':>10s} {'R2 med':>10s} {'L-SHADE mean':>12s}")
    print("-" * 92)
    summary = {}
    for p in PROB_IDS:
        r1_errs = r1["errors"][str(p)]["gdm_on"]          # Round 1, same seeds
        r2_errs = results[p]                               # Round 2, same seeds
        name = stored["problems"][p - 1]
        ls = stored["all_errors"]["L-SHADE"][name]
        med = lambda e: sorted(e)[len(e) // 2]
        pv = wilcoxon(r2_errs, r1_errs)
        print(f"F{p:<4d} {np.mean(r1_errs):>13.4f} {np.mean(r2_errs):>13.4f} {pv:>9.4f} "
              f"{med(r1_errs):>10.4f} {med(r2_errs):>10.4f} {np.mean(ls):>12.4f}")
        summary[p] = {"r1": r1_errs, "r2": r2_errs}
    print("=" * 92)

    out = {
        "prob_ids": PROB_IDS, "n_runs": N_RUNS, "max_iter": MAX_ITER, "dim": DIM,
        "variant": "GDM + Geodesic Re-seeding",
        "errors": {str(p): {"r2": results[p], "r1_ref": r1["errors"][str(p)]["gdm_on"]} for p in PROB_IDS},
    }
    path = os.path.join(os.path.dirname(__file__), "results", "round2_gr_validation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
