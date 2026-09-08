"""CEC Standard Benchmark Suite Implementation with Shifts, Rotations, and Biases.

Implements standard benchmark functions spanning Unimodal, Multimodal,
Hybrid, and Composition categories for testing Evolutionary Algorithms.
"""

from typing import List, Tuple
import numpy as np
from benchmarks.base import BenchmarkFunction


def generate_orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    """Generate a reproducible random orthogonal rotation matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    # Ensure determinant is +1 (proper rotation without reflection)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    Q = Q * ph
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def generate_shift_vector(dim: int, seed: int, bound_scale: float = 80.0) -> np.ndarray:
    """Generate reproducible shift vector within search bounds [-bound_scale, bound_scale]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-bound_scale, bound_scale, size=dim)


# Raw function definitions (evaluated on rotated/shifted z)
def raw_bent_cigar(z: np.ndarray) -> np.ndarray:
    # f(z) = z_1^2 + 10^6 * sum_{i=2}^D z_i^2
    return z[:, 0]**2 + 1e6 * np.sum(z[:, 1:]**2, axis=-1)


def raw_zakharov(z: np.ndarray) -> np.ndarray:
    # f(z) = sum(z_i^2) + (sum(0.5 * i * z_i))^2 + (sum(0.5 * i * z_i))^4
    dim = z.shape[1]
    i_vec = np.arange(1, dim + 1)
    term1 = np.sum(z**2, axis=-1)
    term2 = np.sum(0.5 * i_vec * z, axis=-1)
    return term1 + term2**2 + term2**4


def raw_rosenbrock(z: np.ndarray) -> np.ndarray:
    # f(z) = sum_{i=1}^{D-1} [100 * (z_{i+1} - z_i^2)^2 + (z_i - 1)^2]
    # Note: global optimum at z = 1
    return np.sum(100.0 * (z[:, 1:] - z[:, :-1]**2)**2 + (z[:, :-1] - 1.0)**2, axis=-1)


def raw_rastrigin(z: np.ndarray) -> np.ndarray:
    # f(z) = 10 * D + sum(z_i^2 - 10 * cos(2*pi*z_i))
    dim = z.shape[1]
    return 10.0 * dim + np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z), axis=-1)


def raw_expanded_schaffer_f6(z: np.ndarray) -> np.ndarray:
    # f(z) = g(z_1, z_2) + g(z_2, z_3) + ... + g(z_D, z_1)
    # g(x, y) = 0.5 + (sin^2(sqrt(x^2 + y^2)) - 0.5) / (1 + 0.001*(x^2 + y^2))^2
    z_next = np.roll(z, -1, axis=1)
    r2 = z**2 + z_next**2
    g = 0.5 + (np.sin(np.sqrt(r2))**2 - 0.5) / (1.0 + 0.001 * r2)**2
    return np.sum(g, axis=-1)


def raw_levy(z: np.ndarray) -> np.ndarray:
    # Levy function
    dim = z.shape[1]
    w = 1.0 + (z - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0])**2
    term2 = np.sum((w[:, :-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0)**2), axis=-1)
    term3 = (w[:, -1] - 1.0)**2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1])**2)
    return term1 + term2 + term3


def raw_schwefel(z: np.ndarray) -> np.ndarray:
    # Schwefel 2.26 function
    dim = z.shape[1]
    z_scaled = z + 420.9687462275036  # shifted so optimum is at 0
    return 418.9828872724339 * dim - np.sum(z_scaled * np.sin(np.sqrt(np.abs(z_scaled))), axis=-1)


def raw_griewank(z: np.ndarray) -> np.ndarray:
    # f(z) = 1 + sum(z_i^2 / 4000) - prod(cos(z_i / sqrt(i)))
    dim = z.shape[1]
    i_sqrt = np.sqrt(np.arange(1, dim + 1))
    sum_term = np.sum(z**2 / 4000.0, axis=-1)
    prod_term = np.prod(np.cos(z / i_sqrt), axis=-1)
    return 1.0 + sum_term - prod_term


def raw_hybrid_1(z: np.ndarray) -> np.ndarray:
    # Split variables into 3 groups: Bent Cigar (30%), Rastrigin (30%), Rosenbrock (40%)
    dim = z.shape[1]
    p1 = max(1, int(0.3 * dim))
    p2 = max(p1 + 1, int(0.6 * dim))
    
    z1 = z[:, :p1]
    z2 = z[:, p1:p2]
    z3 = z[:, p2:]
    
    val1 = z1[:, 0]**2 + 1e6 * np.sum(z1[:, 1:]**2, axis=-1) if p1 > 1 else z1[:, 0]**2
    val2 = 10.0 * z2.shape[1] + np.sum(z2**2 - 10.0 * np.cos(2.0 * np.pi * z2), axis=-1)
    val3 = raw_rosenbrock(z3 + 1.0) if z3.shape[1] > 1 else z3[:, 0]**2
    
    return val1 + val2 + val3


def raw_composition_1(z: np.ndarray) -> np.ndarray:
    # Blend 3 basins: Rastrigin, Griewank, Schwefel with Gaussian weights
    dim = z.shape[1]
    n_components = 3
    sigmas = np.array([10.0, 20.0, 30.0])
    biases = np.array([0.0, 100.0, 200.0])
    
    # Centers relative to z
    centers = np.zeros((n_components, dim))
    centers[0] = 0.0
    centers[1] = 10.0
    centers[2] = -10.0
    
    # Evaluate weights
    dist2 = np.zeros((len(z), n_components))
    for i in range(n_components):
        dist2[:, i] = np.sum((z - centers[i])**2, axis=-1)
        
    weights = np.zeros_like(dist2)
    for i in range(n_components):
        weights[:, i] = np.exp(-dist2[:, i] / (2.0 * dim * sigmas[i]**2))
        
    w_sum = np.sum(weights, axis=-1, keepdims=True)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)
    weights /= w_sum
    
    f1 = raw_rastrigin(z - centers[0])
    f2 = raw_griewank(z - centers[1])
    f3 = raw_schwefel(z - centers[2])
    
    comp_val = (
        weights[:, 0] * (f1 + biases[0]) +
        weights[:, 1] * (f2 + biases[1]) +
        weights[:, 2] * (f3 + biases[2])
    )
    
    # Compute offset so value at z=0 is exactly 0
    f1_0 = float(raw_rastrigin(np.zeros((1, dim)))[0])
    f2_0 = float(raw_griewank(np.zeros((1, dim)) - centers[1])[0])
    f3_0 = float(raw_schwefel(np.zeros((1, dim)) - centers[2])[0])
    dist2_0 = np.array([[0.0, float(np.sum(centers[1]**2)), float(np.sum(centers[2]**2))]])
    w_0 = np.zeros_like(dist2_0)
    for i in range(n_components):
        w_0[:, i] = np.exp(-dist2_0[:, i] / (2.0 * dim * sigmas[i]**2))
    w_0 /= np.sum(w_0)
    comp_0 = float(w_0[0, 0] * (f1_0 + biases[0]) + w_0[0, 1] * (f2_0 + biases[1]) + w_0[0, 2] * (f3_0 + biases[2]))
    
    return comp_val - comp_0


class CECBenchmark:
    """Factory for standard CEC benchmark functions."""
    
    @staticmethod
    def get_function(func_id: int, dim: int) -> BenchmarkFunction:
        """Create a benchmark function by ID (1 to 10)."""
        seed_shift = 1000 + func_id * 17
        seed_rot = 2000 + func_id * 31
        
        shift = generate_shift_vector(dim, seed_shift)
        rot = generate_orthogonal_matrix(dim, seed_rot)
        bias = float(func_id * 100.0)
        
        definitions = {
            1: ("Shifted & Rotated Bent Cigar", raw_bent_cigar, "Unimodal"),
            2: ("Shifted & Rotated Zakharov", raw_zakharov, "Unimodal"),
            3: ("Shifted & Rotated Rosenbrock", raw_rosenbrock, "Unimodal/Narrow Valley"),
            4: ("Shifted & Rotated Rastrigin", raw_rastrigin, "Multimodal"),
            5: ("Shifted & Rotated Expanded Schaffer F6", raw_expanded_schaffer_f6, "Multimodal"),
            6: ("Shifted & Rotated Levy", raw_levy, "Multimodal"),
            7: ("Shifted & Rotated Schwefel", raw_schwefel, "Multimodal"),
            8: ("Shifted & Rotated Griewank", raw_griewank, "Multimodal"),
            9: ("Shifted & Rotated Hybrid (Cigar+Rastrigin+Rosenbrock)", raw_hybrid_1, "Hybrid"),
            10: ("Composition Function (Rastrigin+Griewank+Schwefel)", raw_composition_1, "Composition")
        }
        
        if func_id not in definitions:
            raise ValueError(f"Invalid function ID {func_id}. Choose between 1 and 10.")
            
        name, raw_func, category = definitions[func_id]
        
        # Determine global optimum coordinate in decision space
        # For Rosenbrock (F3) and Levy (F6), optimum is at z = 1 => x* = 1 @ rot.T + shift
        if func_id in (3, 6):
            opt_x = np.ones(dim) @ rot.T + shift
        else:
            opt_x = shift.copy()
            
        return BenchmarkFunction(
            name=f"F{func_id}: {name}",
            dim=dim,
            raw_func=raw_func,
            optimum_x=opt_x,
            bias=bias,
            shift_vector=shift,
            rotation_matrix=rot,
            bounds=(-100.0, 100.0),
            category=category
        )


def get_benchmark_suite(dim: int) -> List[BenchmarkFunction]:
    """Retrieve the full 10-function benchmark suite."""
    return [CECBenchmark.get_function(fid, dim) for fid in range(1, 11)]
