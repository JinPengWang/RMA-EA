"""Round-5: escape geometry of the locked wrong basin on F10-D30.

From the failing run's locked point (seed 14520), Monte-Carlo the fraction
of isotropic Gaussian jumps at various scales that land in f < f_lock regions,
plus the geometric distance to the true optimum.
"""
import sys
import numpy as np

sys.path.insert(0, r"E:/A_Works/paper/Evolutionary Algorithm/src")
from benchmarks.cec_suite import get_benchmark_suite  # noqa: E402
from rma_ea.algorithm import RMA_EA  # noqa: E402

DIM, MAX_ITER, SEED = 30, 1500, 14520

f10 = [p for p in get_benchmark_suite(dim=DIM) if "Composition" in p.name][0]
rng = np.random.default_rng(SEED)

# Re-run to capture the locked point
ea = RMA_EA(objective_func=f10, dim=DIM,
            lower_bound=f10.bounds[0], upper_bound=f10.bounds[1],
            max_iter=MAX_ITER, seed=SEED)
res = ea.optimize()
x_lock = res.best_x.copy()
f_lock = res.best_f
print(f"locked point: f = {f_lock:.4f} (err {f_lock - f10.bias:.4f})")

x_star = f10.optimum_x
d_star = float(np.linalg.norm(x_lock - x_star))
print(f"distance to true optimum x*: {d_star:.3f}")
print(f"sigma0 (domain RMS radius):   {ea.sigma0:.3f}")

# z-space location of the locked point (basin identification)
z_lock = (x_lock - f10.shift) @ f10.rotation
c0 = np.zeros(DIM); c1 = 10.0 * np.ones(DIM); c2 = -10.0 * np.ones(DIM)
print("\nz-space distances to basin centers:")
print(f"  ||z - Rastrigin@0||  = {np.linalg.norm(z_lock - c0):.2f}")
print(f"  ||z - Griewank@+10|| = {np.linalg.norm(z_lock - c1):.2f}")
print(f"  ||z - Schwefel@-10|| = {np.linalg.norm(z_lock - c2):.2f}")

# Monte-Carlo escape fractions at candidate diffusion scales
print("\nGaussian jump escape study (n=4000 per scale):")
print(f"{'scale':>8} {'P(f<f_lock-3)':>14} {'P(f<1100)':>12} {'P(f<1050)':>12} {'mean f':>10}")
for s in [0.3, 1.0, 3.0, 10.0, 20.0, 35.0, 50.0, 57.7]:
    jumps = x_lock + s * rng.standard_normal((4000, DIM))
    jumps = np.clip(jumps, f10.bounds[0], f10.bounds[1])
    fj = f10(jumps)
    p1 = np.mean(fj < f_lock - 3.0)
    p2 = np.mean(fj < 1100.0)
    p3 = np.mean(fj < 1050.0)
    print(f"{s:8.1f} {p1:14.4f} {p2:12.4f} {p3:12.4f} {fj.mean():10.1f}")

# Sanity: f at the true optimum
print(f"\nf(x*) = {float(f10(x_star.reshape(1,-1))[0]):.4f} (bias {f10.bias})")
