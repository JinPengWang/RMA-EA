"""Round-5 Validation: Population-Level Atlas Renewal (uniform restart).

Compares RMA-EA v3.4 (chart exhaustion renews the atlas: uniform re-seeding
of the sampler, incumbent retained aside, archive released, cometric reset)
against the stored v3.3 30-run full-benchmark results (identical seed
schedule), on ALL 10 problems at both D10 and D30.

Focus:
  (a) F10-D30 must recover on the two failing seeds (14520, 14521) and not
      lose the 28 successes;
  (b) no regression on any other problem (renewal must be inert where the
      atlas is never exhausted, and harmless where exhaustion fires late).

Protocol: seed = 10000 + (prob_id-1)*500 + run_idx; max_iter 1000 (D10) /
1500 (D30); error threshold 1e-8.
"""

import sys
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA
from benchmarks.cec_suite import get_benchmark_suite

PROB_IDS = list(range(1, 11))
N_RUNS = 30


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
            results.setdefault((dim, prob_id), []).append((run_idx, err))
            if (i + 1) % 30 == 0:
                print(f"[{i + 1}/{len(tasks)}] done", flush=True)

    # order by run_idx for comparability
    for k in results:
        results[k].sort()
        results[k] = [e for _, e in results[k]]

    refs = {}
    for dim in (10, 30):
        with open(os.path.join(os.path.dirname(__file__), "results", f"cec_results_D{dim}.json")) as fh:
            refs[dim] = json.load(fh)

    print("\n" + "=" * 100)
    print(f"{'Case':9s} {'v3.3 mean':>12s} {'v3.4 mean':>12s} {'p(v3.4~v3.3)':>13s} "
          f"{'v3.3 #fail':>10s} {'v3.4 #fail':>10s} {'L-SHADE mean':>12s}")
    print("-" * 100)
    regressions = []
    for dim in (10, 30):
        for p in PROB_IDS:
            name = refs[dim]["problems"][p - 1]
            old = refs[dim]["all_errors"]["RMA-EA"][name]
            new = results[(dim, p)]
            ls = refs[dim]["all_errors"]["L-SHADE"][name]
            pv = wilcoxon(new, old)
            nfail_o = sum(1 for e in old if e > 1e-8)
            nfail_n = sum(1 for e in new if e > 1e-8)
            mark = ""
            if np.mean(new) > np.mean(old) * 1.05 + 1e-12 and np.mean(new) > 1e-8:
                mark = "  [CHECK]"
                regressions.append((dim, p))
            print(f"D{dim} F{p:<3d} {np.mean(old):>12.4f} {np.mean(new):>12.4f} {pv:>13.4f} "
                  f"{nfail_o:>10d} {nfail_n:>10d} {np.mean(ls):>12.4f}{mark}")
    print("=" * 100)
    if regressions:
        print("REGRESSION CANDIDATES:", regressions)
    else:
        print("No mean regressions detected.")

    # F10-D30 seed-level comparison
    name10 = refs[30]["problems"][9]
    old_f10 = refs[30]["all_errors"]["RMA-EA"][name10]
    new_f10 = results[(30, 10)]
    print("\nF10-D30 per-seed change (first 30 runs):")
    for r in range(N_RUNS):
        if old_f10[r] != new_f10[r]:
            print(f"  run {r}: {old_f10[r]:.4e} -> {new_f10[r]:.4e}")

    out = {
        "prob_ids": PROB_IDS, "n_runs": N_RUNS,
        "variant": "Population-level atlas renewal (uniform restart)",
        "errors": {f"D{dim}_F{p}": results[(dim, p)] for dim in (10, 30) for p in PROB_IDS},
    }
    path = os.path.join(os.path.dirname(__file__), "results", "round5_renewal_validation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
