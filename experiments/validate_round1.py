"""Round-1 Validation: Geodesic Diffusion Mutation (GDM) A/B Experiment.

Protocol identical to run_cec_experiments.py: dim=10, max_iter=1000, same seed
schedule. Compares RMA-EA with GDM enabled (default) against the pre-Round-1
behavior (geodesic_diffusion=False) on weak multimodal problems and unimodal
sanity checks. L-SHADE reference medians from cec_results_D10.json.
"""

import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA
from benchmarks.cec_suite import get_benchmark_suite

PROB_IDS = [1, 3, 4, 5, 9, 10]   # F1/F3 sanity, F4/F5/F9/F10 weak multimodal
N_RUNS = 10
MAX_ITER = 1000
DIM = 10


def _worker(args):
    prob_id, run_idx, gdm = args
    suite = get_benchmark_suite(dim=DIM)
    func = suite[prob_id - 1]
    seed = 10000 + (prob_id - 1) * 500 + run_idx  # same schedule as main protocol
    opt = RMA_EA(
        objective_func=func,
        dim=DIM,
        lower_bound=func.bounds[0],
        upper_bound=func.bounds[1],
        max_iter=MAX_ITER,
        geodesic_diffusion=gdm,
        seed=seed,
    )
    res = opt.optimize()
    err = max(0.0, float(res.best_f - func.bias))
    return prob_id, run_idx, gdm, 0.0 if err < 1e-8 else err


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
    tasks = [(p, r, g) for p in PROB_IDS for r in range(N_RUNS) for g in (True, False)]
    results = {p: {True: [], False: []} for p in PROB_IDS}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            prob_id, run_idx, gdm, err = f.result()
            results[prob_id][gdm].append(err)
            print(f"[{i + 1}/{len(tasks)}] F{prob_id} run{run_idx} GDM={gdm}: {err:.4e}")

    # L-SHADE reference medians from the stored D10 benchmark results
    with open(os.path.join(os.path.dirname(__file__), "results", "cec_results_D10.json")) as fh:
        stored = json.load(fh)
    lshade_med = {}
    for p in PROB_IDS:
        name = stored["problems"][p - 1]
        errs = sorted(stored["all_errors"]["L-SHADE"][name])
        lshade_med[p] = errs[len(errs) // 2]

    print("\n" + "=" * 78)
    print(f"{'Prob':5s} {'GDM=OFF median':>16s} {'GDM=ON median':>16s} {'L-SHADE ref':>14s}  {'Wilcoxon p':>10s}")
    print("-" * 78)
    for p in PROB_IDS:
        off, on = sorted(results[p][False]), sorted(results[p][True])
        med_off, med_on = off[len(off) // 2], on[len(on) // 2]
        pv = wilcoxon(results[p][True], results[p][False])
        print(f"F{p:<4d} {med_off:>16.4e} {med_on:>16.4e} {lshade_med[p]:>14.4e}  {pv:>10.4f}")
    print("=" * 78)

    out = {
        "prob_ids": PROB_IDS, "n_runs": N_RUNS, "max_iter": MAX_ITER, "dim": DIM,
        "errors": {str(p): {"gdm_on": results[p][True], "gdm_off": results[p][False]} for p in PROB_IDS},
        "lshade_median_ref": {str(p): lshade_med[p] for p in PROB_IDS},
    }
    path = os.path.join(os.path.dirname(__file__), "results", "round1_gdm_validation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
