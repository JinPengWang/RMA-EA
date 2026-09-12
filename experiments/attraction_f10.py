"""Round-5: verify the attraction structure around the true optimum.

The trapped population sits 268 away from x*. Direct landings in
{f < f_lock} fail (0/4000). But 28/30 runs succeed via DESCENT - so the
ATTRACTION region of the good basin must be much larger than the
{f < f_lock} region. Quantify both to justify population-level renewal.
"""
import sys
import numpy as np

sys.path.insert(0, r"E:/A_Works/paper/Evolutionary Algorithm/src")
from benchmarks.cec_suite import get_benchmark_suite  # noqa: E402

DIM = 30
f10 = [p for p in get_benchmark_suite(dim=DIM) if "Composition" in p.name][0]
rng = np.random.default_rng(7)
x_star = f10.optimum_x
f_lock = 1132.5699

# 1. Region where f < f_lock around x* (direct-landing target)
print("Direct-landing region {f < 1132.57} around x*:")
for r in [5, 10, 20, 30, 50, 80]:
    pts = x_star + r * rng.standard_normal((2000, DIM))
    pts = np.clip(pts, f10.bounds[0], f10.bounds[1])
    frac = np.mean(f10(pts) < f_lock)
    print(f"  radius {r:3d} (RMS): P(f < f_lock) = {frac:.4f}")

# 2. Descent reachability: random domain points descend below f_lock?
#    Proxy: fraction of RANDOM domain points whose immediate neighborhood
#    has a downhill path toward x* -- estimated by local gradient sign.
#    Simpler proxy: fraction of random points with f < f_lock at all.
pts = rng.uniform(f10.bounds[0], f10.bounds[1], size=(4000, DIM))
fvals = f10(pts)
print(f"\nRandom domain points: P(f < f_lock) = {np.mean(fvals < f_lock):.4f}, mean f = {fvals.mean():.1f}")

# 3. The key mechanism: in fresh runs the population DESCENDS into the good
#    basin. Test: short greedy descent (hill climbing with line searches is
#    too expensive; use coordinate-wise descent probe) from random points is
#    approximated by sampling toward x*: points along rays toward x*.
print("\nAlong-ray values from a random point toward x* (does f decrease monotonically?)")
x0 = rng.uniform(f10.bounds[0], f10.bounds[1], size=DIM)
for t in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    xt = x0 + t * (x_star - x0)
    print(f"  t={t:4.2f}: f = {float(f10(xt.reshape(1,-1))[0]):10.1f}")
