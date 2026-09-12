"""Riemannian Manifold Engine for RMA-EA.

Implements operations on the Symmetric Positive Definite (SPD) manifold
under the Log-Euclidean Riemannian metric.
"""

from typing import Tuple, Optional
import numpy as np


def symmetrize(M: np.ndarray) -> np.ndarray:
    """Enforce exact numerical symmetry."""
    return 0.5 * (M + M.T)


def matrix_log(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute matrix logarithm on SPD manifold.
    
    log(C) = U * diag(ln(lambda)) * U.T
    """
    C_sym = symmetrize(C)
    eig_vals, eig_vecs = np.linalg.eigh(C_sym)
    eig_vals = np.maximum(eig_vals, eps)
    log_vals = np.log(eig_vals)
    log_C = eig_vecs @ np.diag(log_vals) @ eig_vecs.T
    return symmetrize(log_C)


def matrix_exp(M: np.ndarray, max_val: float = 50.0) -> np.ndarray:
    """Compute matrix exponential mapping from symmetric space to SPD manifold.
    
    exp(M) = U * diag(exp(lambda)) * U.T
    """
    M_sym = symmetrize(M)
    eig_vals, eig_vecs = np.linalg.eigh(M_sym)
    eig_vals = np.clip(eig_vals, -max_val, max_val)
    exp_vals = np.exp(eig_vals)
    exp_M = eig_vecs @ np.diag(exp_vals) @ eig_vecs.T
    return symmetrize(exp_M)


def log_euclidean_distance(C1: np.ndarray, C2: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Log-Euclidean Riemannian distance between two SPD matrices.
    
    d_LE(C1, C2) = ||log(C1) - log(C2)||_F
    """
    log_C1 = matrix_log(C1, eps=eps)
    log_C2 = matrix_log(C2, eps=eps)
    diff = log_C1 - log_C2
    return float(np.linalg.norm(diff, ord='fro'))


def geodesic(C1: np.ndarray, C2: np.ndarray, tau: float, eps: float = 1e-12) -> np.ndarray:
    """Compute geodesic curve gamma(tau) connecting C1 and C2 on SPD manifold.
    
    gamma(tau) = exp( (1 - tau) * log(C1) + tau * log(C2) )
    tau in [0, 1]
    """
    tau = np.clip(tau, 0.0, 1.0)
    log_C1 = matrix_log(C1, eps=eps)
    log_C2 = matrix_log(C2, eps=eps)
    log_interp = (1.0 - tau) * log_C1 + tau * log_C2
    return matrix_exp(log_interp)


def update_riemannian_barycenter(
    C_prev: np.ndarray,
    C_new: np.ndarray,
    learning_rate: float,
    eps: float = 1e-12
) -> np.ndarray:
    """Online update of Riemannian barycenter (geometric mean) on SPD manifold.
    
    log(C_updated) = (1 - lr) * log(C_prev) + lr * log(C_new)
    """
    learning_rate = float(np.clip(learning_rate, 0.0, 1.0))
    if learning_rate <= 0.0:
        return symmetrize(C_prev.copy())
    if learning_rate >= 1.0:
        return symmetrize(C_new.copy())
    
    log_prev = matrix_log(C_prev, eps=eps)
    log_new = matrix_log(C_new, eps=eps)
    log_updated = (1.0 - learning_rate) * log_prev + learning_rate * log_new
    return matrix_exp(log_updated)


class RiemannianMetricFlow:
    """Continuous Riemannian Metric Flow Tracker on S++^D.
    
    Maintains the empirical cometric tensor C_t = G_t^{-1} along the manifold
    of positive-definite matrices:
        C_{t+1} = (1 - c_c) C_t + c_c * (C_emp / Tr(C_emp)/D)
    Guarantees full-rank Riemannian geometry across all search phases even
    when the population size N drops below problem dimension D.
    """
    def __init__(self, dim: int, learning_rate: Optional[float] = None):
        self.dim = dim
        self.c_c_inf = learning_rate if learning_rate is not None else 2.0 / (dim ** 1.5)
        self.C = np.eye(dim, dtype=np.float64)
        self.U = np.eye(dim, dtype=np.float64)
        self.eig_vals = np.ones(dim, dtype=np.float64)
        self.iteration = 0

    def reset(self) -> None:
        """Renew the chart: reset the cometric tensor to the isotropic metric.

        Invoked when the current chart is exhausted (prolonged stagnation), so
        that subsequent updates restart with the Robbins-Monro warm-up.
        """
        self.C = np.eye(self.dim, dtype=np.float64)
        self.U = np.eye(self.dim, dtype=np.float64)
        self.eig_vals = np.ones(self.dim, dtype=np.float64)
        self.iteration = 0
        
    def update(self, elite_samples: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Update Riemannian metric flow with newly observed elite individuals.

        Args:
            elite_samples: (mu, D) matrix of elite decision vectors.
            weights: (mu,) normalized ranking weights (sum(w) = 1).

        Returns:
            U: (D, D) orthonormal Riemannian tangent basis.
            eig_vals: (D,) sorted eigenvalues of the cometric tensor.
            sigma_pop: current elite spread sqrt(Tr(C_emp)/D), the physical scale
                of the population for scale-referenced geodesic diffusion.
        """
        self.iteration += 1
        mu, D = elite_samples.shape
        w_sum = np.sum(weights)
        norm_w = weights / w_sum if w_sum > 0 else np.ones(mu) / mu

        # Weighted mean and empirical covariance
        mean = np.sum(elite_samples * norm_w[:, np.newaxis], axis=0)
        diff = elite_samples - mean
        C_emp = (diff.T * norm_w) @ diff

        # Scale-invariant normalization; retain the physical scale sigma_pop
        tr_mean = np.trace(C_emp) / max(D, 1)
        sigma_pop = float(np.sqrt(max(tr_mean, 1e-24)))
        C_norm = C_emp / (tr_mean + 1e-12) if tr_mean > 1e-12 else np.eye(D)

        # Parameter-free Bayesian Riemannian shrinkage towards isotropic metric I
        # rho = D / (mu + D) prevents Marchenko-Pastur rank-deficiency when mu < D.
        # The full-rank floor is not only estimation robustness: it keeps the
        # geodesic diffusion rank-complete in ALL D directions, which Round-3
        # ablation showed is the exploration signal itself (confining diffusion
        # to the raw elite subspace collapses multimodal escape).
        rho = float(D) / float(mu + D)
        C_shrunk = (1.0 - rho) * C_norm + rho * np.eye(D)
        
        # Bayesian optimal metric flow integration (warm-up decaying to asymptotic rate)
        c_c = max(2.0 / (self.iteration + 2.0), self.c_c_inf)
        self.C = (1.0 - c_c) * self.C + c_c * C_shrunk
        self.C = symmetrize(self.C)
        
        # Spectral decomposition of cometric tensor
        vals, vecs = np.linalg.eigh(self.C)
        sort_idx = np.argsort(vals)[::-1]
        self.eig_vals = np.maximum(vals[sort_idx], 1e-12)
        self.U = vecs[:, sort_idx]

        return self.U, self.eig_vals, sigma_pop


def low_rank_covariance_decompose(
    samples: np.ndarray,
    weights: np.ndarray,
    rank_k: int,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute low-rank decomposition of empirical covariance matrix.
    
    Returns:
        eig_vals: (k,) top k eigenvalues
        eig_vecs: (D, k) top k orthonormal eigenvectors
        sigma_res: scalar residual standard deviation for remaining D - k dimensions
    """
    n_samples, dim = samples.shape
    rank_k = min(rank_k, dim, n_samples - 1)
    rank_k = max(1, rank_k)
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 0:
        norm_weights = np.ones(n_samples) / n_samples
    else:
        norm_weights = weights / w_sum
        
    # Weighted mean
    mean = np.sum(samples * norm_weights[:, np.newaxis], axis=0)
    diff = samples - mean
    
    # Weighted covariance matrix
    cov = (diff.T * norm_weights) @ diff + np.eye(dim) * eps
    cov = symmetrize(cov)
    
    # Spectral decomposition
    all_eig_vals, all_eig_vecs = np.linalg.eigh(cov)
    
    # Sort descending
    idx = np.argsort(all_eig_vals)[::-1]
    sorted_vals = np.maximum(all_eig_vals[idx], eps)
    sorted_vecs = all_eig_vecs[:, idx]
    
    top_vals = sorted_vals[:rank_k]
    top_vecs = sorted_vecs[:, :rank_k]
    
    if dim > rank_k:
        residual_variance = np.mean(sorted_vals[rank_k:])
        sigma_res = float(np.sqrt(max(residual_variance, eps)))
    else:
        sigma_res = float(np.sqrt(eps))
        
    return top_vals, top_vecs, sigma_res


def sample_geodesic_perturbation(
    eig_vecs: np.ndarray,
    eig_vals: np.ndarray,
    sigma_res: float,
    size: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Sample anisotropic perturbations aligned with the Riemannian principal directions.
    
    Delta = sum_{j=1}^k sqrt(lambda_j) * z_j * v_j + sigma_res * xi_res
    """
    if rng is None:
        rng = np.random.default_rng()
        
    dim, rank_k = eig_vecs.shape
    # Principal components
    z_principal = rng.standard_normal((size, rank_k))  # (size, k)
    principal_scaled = z_principal * np.sqrt(eig_vals)  # (size, k)
    delta_principal = principal_scaled @ eig_vecs.T    # (size, dim)
    
    return delta_principal


def project_to_eigen_basis(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Project points from Cartesian coordinates to Riemannian eigen-basis.
    
    Y = X @ B
    """
    return X @ B


def reproject_from_eigen_basis(Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reproject points from Riemannian eigen-basis back to Cartesian coordinates.
    
    X = Y @ B.T
    """
    return Y @ B.T
