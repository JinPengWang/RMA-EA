"""Round-5: instrumented reproduction of the F10-D30 failing seeds (10470, 10471).

Records per-iteration: global best fitness, population spread, sigma_pop,
cometric condition number, p_rot / p_diff posteriors, and geodesic reseed
events, to locate when/why the run locks into the wrong composition basin.
"""
import sys
import numpy as np

sys.path.insert(0, r"E:/A_Works/paper/Evolutionary Algorithm/src")
from benchmarks.cec_suite import get_benchmark_suite  # noqa: E402
from rma_ea.algorithm import RMA_EA  # noqa: E402

SEEDS = [14520, 14521]
DIM, MAX_ITER = 30, 1500

probs = get_benchmark_suite(dim=DIM)
f10 = [p for p in probs if "Composition" in p.name][0]
print("F10:", f10.name, "bounds:", f10.bounds, "bias:", f10.bias)
print("optimum_x == shift:", np.allclose(f10.optimum_x, f10.shift))


class MetricFlowProbe:
    """Delegating proxy logging metric-flow state each iteration."""

    def __init__(self, inner):
        self._inner = inner
        self.log = []          # (iter, sigma_pop, cond)
        self.resets = []       # iteration numbers of reset()
        self._it = 0

    def update(self, *a, **kw):
        out = self._inner.update(*a, **kw)
        _, eig_vals, sigma_pop = out
        cond = float(eig_vals[0] / max(eig_vals[-1], 1e-300))
        self._it += 1
        self.log.append((self._it, float(sigma_pop), cond))
        return out

    def reset(self):
        self.resets.append(self._it)
        return self._inner.reset()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MemoryProbe:
    """Delegating proxy logging p_rot/p_diff per generation."""

    def __init__(self, inner):
        self._inner = inner
        self.log = []          # (gen, p_rot, p_diff)

    def sample_parameters(self, *a, **kw):
        out = self._inner.sample_parameters(*a, **kw)
        self.log.append((len(self.log) + 1, float(self._inner.p_rot), float(self._inner.p_diff)))
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"SEED {seed}  (F10 D{DIM})")
    print("=" * 70)
    ea = RMA_EA(
        objective_func=f10,
        dim=DIM,
        lower_bound=f10.bounds[0],
        upper_bound=f10.bounds[1],
        max_iter=MAX_ITER,
        seed=seed,
    )
    mfp = MetricFlowProbe(ea.metric_flow)
    mmp = MemoryProbe(ea.memory)
    ea.metric_flow = mfp
    ea.memory = mmp

    res = ea.optimize()
    err = res.best_f - f10.bias
    print(f"final best_f={res.best_f:.6e}  err={err:.4e}  iters={res.iterations}")

    hf = np.array(res.history_fitness)
    # locate plateau: first iteration where best stays within 1e-6 of final
    plateau_from = int(np.argmax(np.abs(hf - hf[-1]) < 1e-6)) if np.any(np.abs(hf - hf[-1]) < 1e-6) else -1
    print(f"plateau (|f-final|<1e-6) starts at iter {plateau_from}/{len(hf)-1}")
    step = max(1, (len(hf)) // 15)
    print("best-f trajectory:")
    for i in range(0, len(hf), step):
        print(f"   it {i:5d}: {hf[i]:.4e}")
    print(f"   it {len(hf)-1:5d}: {hf[-1]:.4e}")

    print(f"reseed events ({len(mfp.resets)}):", mfp.resets[:20], "..." if len(mfp.resets) > 20 else "")

    # internal state at a few key iterations
    ml = mfp.log
    print("metric-flow state (iter, sigma_pop, cond):")
    for i in [1, 50, 100, 200, 300, 500, 800, 1100, 1400, len(ml)]:
        if 1 <= i <= len(ml):
            it, sp, cond = ml[i - 1]
            print(f"   it {it:5d}: sigma_pop={sp:.3e}  cond={cond:.3e}")
    print("p-posteriors (gen, p_rot, p_diff):")
    pl = mmp.log
    for i in [1, 20, 50, 100, 200, 400, 800, 1200, len(pl)]:
        if 1 <= i <= len(pl):
            g, pr, pd = pl[i - 1]
            print(f"   gen {g:5d}: p_rot={pr:.3f}  p_diff={pd:.3f}")
