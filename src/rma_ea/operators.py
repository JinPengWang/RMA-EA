"""Evolutionary Operators and Adaptive Parameter Memory for RMA-EA (Version 2.0).

Implements:
1. Dual-channel geodesic mutation with asymptotic decay.
2. Riemannian Eigen-Coordinate Crossover (rotation-invariant).
3. Parameter Memory with Lehmer updates.
4. Midpoint bound repair.
"""

from typing import Tuple, Optional, Union
import numpy as np
from .manifold import sample_geodesic_perturbation, project_to_eigen_basis, reproject_from_eigen_basis


def repair_bounds(
    x: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    """Repair out-of-bound variables using midpoint projection to prevent edge stagnation.
    
    If x_j < lower_j: x_j = (target_j + lower_j) / 2
    If x_j > upper_j: x_j = (target_j + upper_j) / 2
    """
    repaired = x.copy()
    lower_b = lower[np.newaxis, :] if lower.ndim == 1 and repaired.ndim == 2 else lower
    upper_b = upper[np.newaxis, :] if upper.ndim == 1 and repaired.ndim == 2 else upper
    
    mask_low = repaired < lower_b
    mask_high = repaired > upper_b
    
    repaired[mask_low] = (target[mask_low] + np.broadcast_to(lower_b, repaired.shape)[mask_low]) / 2.0
    repaired[mask_high] = (target[mask_high] + np.broadcast_to(upper_b, repaired.shape)[mask_high]) / 2.0
    
    return np.clip(repaired, lower_b, upper_b)


class ParameterMemory:
    """Riemannian Atlas Parameter Memory with Operator-Decoupled Posteriors.

    In Riemannian Differential Geometry, a manifold atlas A = {(U_alpha, phi_alpha)}
    consists of coordinate charts with distinct characteristic velocities.
    This class maintains chart-specific historical memory:
        Chart 0 (Canonical Euclidean chart phi_0): (M_F_can, M_Cr_can)
        Chart 1 (Riemannian Principal Tangent chart phi_1): (M_F_rot, M_Cr_rot)
    Eliminates parameter cross-contamination across coordinate charts.

    The two operators that consume the atlas receive SEPARATE Bayesian chart
    posteriors, each credited by the signal appropriate to its role:
    1. Crossover chart posterior p_rot (mode selection quality):
       success-count credit, Laplace-regularized survival rate.
           r_rot = (n_rot_succ + 0.1) / (n_rot_eval + 0.2)
           p_rot <- (1 - c_chart) p_rot + c_chart r_rot / (r_rot + r_can)
    2. Diffusion chart posterior p_diff (exploration productivity):
       improvement-MASS credit per evaluation. The target is a ratio of same
       units and therefore scale-invariant across problems; it demotes the
       geodesic diffusion exactly where it yields no displacement quality
       (smooth blocks of hybrid landscapes) and promotes it where hops
       deliver large improvements (multimodal basins).
           r_diff = d_diff / (n_diff_eval + 1)
           p_diff <- (1 - c_chart) p_diff + c_chart r_diff / (r_diff + r_can)
    Both smoothing rates equal c_chart = 1 / H.
    """
    def __init__(self, memory_size: int = 6, init_F: float = 0.5, init_Cr: float = 0.5):
        self.memory_size = memory_size
        self.c_chart = 1.0 / float(memory_size)

        # Chart 0: Canonical coordinate chart memory
        self.M_F_can = np.full(memory_size, init_F, dtype=np.float64)
        self.M_Cr_can = np.full(memory_size, init_Cr, dtype=np.float64)
        self.ptr_can = 0

        # Chart 1: Riemannian principal tangent chart memory
        self.M_F_rot = np.full(memory_size, init_F, dtype=np.float64)
        self.M_Cr_rot = np.full(memory_size, init_Cr, dtype=np.float64)
        self.ptr_rot = 0

        # Operator-decoupled Bayesian chart posteriors
        self.p_rot = 0.5    # crossover chart (count credit)
        self.p_diff = 0.5  # diffusion mutation chart (mass credit)

        # Backward-compatibility properties
        self.M_F = self.M_F_can
        self.M_Cr = self.M_Cr_can
        self.M_chart = np.full(memory_size, 0.5, dtype=np.float64)
        self.memory_ptr = 0
        
    def sample_parameters(
        self,
        size: int,
        rng: Optional[np.random.Generator] = None,
        return_chart: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Sample chart-specific F and Cr parameters according to the Riemannian atlas.

        Returns (with return_chart=True) the crossover-chart mask and the
        diffusion-chart mask, drawn independently from their respective
        operator posteriors p_rot and p_diff.
        """
        if rng is None:
            rng = np.random.default_rng()

        use_chart = rng.random(size) < self.p_rot
        use_diff = rng.random(size) < self.p_diff
        F = np.zeros(size, dtype=np.float64)
        Cr = np.zeros(size, dtype=np.float64)
        slot_indices = np.zeros(size, dtype=int)
        
        # Sample for Chart 0 (Canonical)
        idx_can = np.where(~use_chart)[0]
        if len(idx_can) > 0:
            r_c = rng.integers(0, self.memory_size, size=len(idx_can))
            slot_indices[idx_can] = r_c
            for i_c, i_pop in enumerate(idx_can):
                while True:
                    val = self.M_F_can[r_c[i_c]] + 0.1 * rng.standard_cauchy()
                    if val > 1.0:
                        F[i_pop] = 1.0
                        break
                    elif val > 0.0:
                        F[i_pop] = val
                        break
            Cr[idx_can] = np.clip(self.M_Cr_can[r_c] + 0.1 * rng.standard_normal(len(idx_can)), 0.0, 1.0)
            
        # Sample for Chart 1 (Riemannian Tangent)
        idx_rot = np.where(use_chart)[0]
        if len(idx_rot) > 0:
            r_r = rng.integers(0, self.memory_size, size=len(idx_rot))
            slot_indices[idx_rot] = r_r
            for i_r, i_pop in enumerate(idx_rot):
                while True:
                    val = self.M_F_rot[r_r[i_r]] + 0.1 * rng.standard_cauchy()
                    if val > 1.0:
                        F[i_pop] = 1.0
                        break
                    elif val > 0.0:
                        F[i_pop] = val
                        break
            Cr[idx_rot] = np.clip(self.M_Cr_rot[r_r] + 0.1 * rng.standard_normal(len(idx_rot)), 0.0, 1.0)
            
        if return_chart:
            return F, Cr, use_chart, use_diff, slot_indices
        return F, Cr
        
    def update_memory(
        self,
        successful_F: np.ndarray,
        successful_Cr: np.ndarray,
        fitness_improvements: np.ndarray,
        successful_chart: Optional[np.ndarray] = None,
        chart_stats: Optional[Tuple[int, int, int, int]] = None,
        diff_stats: Optional[Tuple[int, int, float, float]] = None
    ) -> None:
        """Update chart-specific memory slots and the two chart posteriors.

        Args:
            chart_stats: (n_rot_eval, n_rot_succ, n_can_eval, n_can_succ) --
                crossover-chart survival counts for the count-credit posterior.
            diff_stats: (n_diff_eval, n_can_eval, d_diff, d_can) -- evaluation
                counts and improvement masses for the mass-credit posterior.
        """
        if len(successful_F) == 0:
            return
            
        if successful_chart is not None:
            succ_can = ~successful_chart
            succ_rot = successful_chart
            
            # Update Chart 0 (Canonical)
            if np.any(succ_can):
                w_c = fitness_improvements[succ_can]
                total_w_c = np.sum(w_c)
                norm_w_c = w_c / total_w_c if total_w_c > 0 else np.ones(len(w_c)) / len(w_c)
                f_c = successful_F[succ_can]
                cr_c = successful_Cr[succ_can]
                mean_L_F_c = np.sum(norm_w_c * (f_c**2)) / max(np.sum(norm_w_c * f_c), 1e-12)
                mean_A_Cr_c = np.sum(norm_w_c * cr_c)
                self.M_F_can[self.ptr_can] = float(np.clip(mean_L_F_c, 0.01, 1.0))
                self.M_Cr_can[self.ptr_can] = float(np.clip(mean_A_Cr_c, 0.0, 1.0))
                self.ptr_can = (self.ptr_can + 1) % self.memory_size
                
            # Update Chart 1 (Riemannian Tangent)
            if np.any(succ_rot):
                w_r = fitness_improvements[succ_rot]
                total_w_r = np.sum(w_r)
                norm_w_r = w_r / total_w_r if total_w_r > 0 else np.ones(len(w_r)) / len(w_r)
                f_r = successful_F[succ_rot]
                cr_r = successful_Cr[succ_rot]
                mean_L_F_r = np.sum(norm_w_r * (f_r**2)) / max(np.sum(norm_w_r * f_r), 1e-12)
                mean_A_Cr_r = np.sum(norm_w_r * cr_r)
                self.M_F_rot[self.ptr_rot] = float(np.clip(mean_L_F_r, 0.01, 1.0))
                self.M_Cr_rot[self.ptr_rot] = float(np.clip(mean_A_Cr_r, 0.0, 1.0))
                self.ptr_rot = (self.ptr_rot + 1) % self.memory_size
        else:
            total_imp = np.sum(fitness_improvements)
            weights = fitness_improvements / total_imp if total_imp > 0 else np.ones(len(fitness_improvements)) / len(fitness_improvements)
            mean_L_F = np.sum(weights * (successful_F**2)) / max(np.sum(weights * successful_F), 1e-12)
            mean_A_Cr = np.sum(weights * successful_Cr)
            self.M_F_can[self.ptr_can] = float(np.clip(mean_L_F, 0.01, 1.0))
            self.M_Cr_can[self.ptr_can] = float(np.clip(mean_A_Cr, 0.0, 1.0))
            self.ptr_can = (self.ptr_can + 1) % self.memory_size

        # Crossover chart posterior: count credit (mode-selection quality)
        if chart_stats is not None:
            n_rot_eval, n_rot_succ, n_can_eval, n_can_succ = chart_stats
            r_rot = (n_rot_succ + 0.1) / (n_rot_eval + 0.2)
            r_can = (n_can_succ + 0.1) / (n_can_eval + 0.2)
            target_p = r_rot / (r_rot + r_can)
            self.p_rot = float(np.clip(
                (1.0 - self.c_chart) * self.p_rot + self.c_chart * target_p,
                0.05, 0.95
            ))
            self.M_chart[:] = self.p_rot

        # Diffusion chart posterior: improvement-mass credit per evaluation.
        # The target is a ratio of same units, hence scale-invariant; the
        # Laplace denominator (one barren evaluation) regularizes the ratio.
        if diff_stats is not None:
            n_diff_eval, n_can_eval, d_diff, d_can = diff_stats
            if d_diff + d_can > 0.0:
                r_diff = d_diff / (n_diff_eval + 1.0)
                r_can = d_can / (n_can_eval + 1.0)
                target_p = r_diff / (r_diff + r_can)
                self.p_diff = float(np.clip(
                    (1.0 - self.c_chart) * self.p_diff + self.c_chart * target_p,
                    0.05, 0.95
                ))
            
        self.memory_ptr = self.ptr_can


class DualChannelMutation:
    """Dual-channel mutation with manifold alignment and asymptotic decay."""
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def mutate(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        pbest_indices: np.ndarray,
        archive: Optional[np.ndarray],
        F: np.ndarray,
        channel_prob: float,
        eig_vecs: np.ndarray,
        eig_vals: np.ndarray,
        sigma_res: float,
        decay_factor: float = 1.0,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Generate donor vectors via current-to-pbest/1 with manifold guidance."""
        if rng is None:
            rng = np.random.default_rng()
            
        n_pop, dim = pop.shape
        donors = np.zeros_like(pop)
        
        # Archive pool
        if archive is not None and len(archive) > 0:
            union_pool = np.vstack([pop, archive])
        else:
            union_pool = pop
        n_union = len(union_pool)
        
        # Sample geodesic perturbation along manifold directions
        geodesic_pert = sample_geodesic_perturbation(
            eig_vecs, eig_vals, sigma_res, size=n_pop, rng=rng
        )
        
        for i in range(n_pop):
            # Select p-best individual
            if hasattr(pbest_indices, '__len__') and len(pbest_indices) == n_pop and isinstance(pbest_indices[0], (int, np.integer)):
                pbest_idx = pbest_indices[i]
            elif hasattr(pbest_indices, '__len__') and len(pbest_indices) > 0:
                pbest_idx = rng.choice(pbest_indices)
            else:
                pbest_idx = rng.integers(0, max(2, int(0.1 * n_pop)))
            x_pbest = pop[pbest_idx]
            
            # Select r1 from pop != i
            r1 = rng.integers(0, n_pop - 1)
            if r1 >= i:
                r1 += 1
            x_r1 = pop[r1]
            
            # Select r2 from union_pool != r1 and != i
            if n_union > 2:
                while True:
                    r2 = rng.integers(0, n_union)
                    if r2 != i and r2 != r1:
                        break
            elif n_union == 2:
                r2 = 1 if (i == 0 or r1 == 0) else 0
            else:
                r2 = 0
            x_r2 = union_pool[r2]
            
            f_i = F[i]
            
            # Core current-to-pbest direction
            base_diff = pop[i] + f_i * (x_pbest - pop[i]) + f_i * (x_r1 - x_r2)
            
            # Dynamic Channel decision
            if rng.random() < channel_prob:
                # Channel A: Exploration along geodesic with asymptotic decay
                pert = 0.02 * f_i * decay_factor * geodesic_pert[i]
                donor = base_diff + pert
            else:
                # Channel B: Pure current-to-pbest exploitation
                donor = base_diff
                
            donors[i] = donor
            
        return donors


def binomial_crossover(
    target: np.ndarray,
    donor: np.ndarray,
    Cr: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Standard Cartesian binomial crossover."""
    if rng is None:
        rng = np.random.default_rng()
        
    n_pop, dim = target.shape
    trial = target.copy()
    
    rand_matrix = rng.random((n_pop, dim))
    mask = rand_matrix <= Cr[:, np.newaxis]
    
    # Guarantee at least 1 donor dimension
    j_rand = rng.integers(0, dim, size=n_pop)
    for i in range(n_pop):
        mask[i, j_rand[i]] = True
        
    trial[mask] = donor[mask]
    return trial


def riemannian_eigen_crossover(
    target: np.ndarray,
    donor: np.ndarray,
    Cr: np.ndarray,
    eigen_basis: Optional[np.ndarray],
    rot_prob: float = 0.8,
    return_used_mask: bool = False,
    rng: Optional[np.random.Generator] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Rotation-invariant crossover performed in the Riemannian eigen-coordinate system.
    
    Performs per-individual adaptive operator selection: individuals selected for
    eigen-crossover are rotated into Riemannian coordinates, while others perform
    standard Cartesian crossover.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n_pop, dim = target.shape
    trial = np.zeros_like(target)
    
    if eigen_basis is not None and eigen_basis.shape == (dim, dim) and rot_prob > 0.0:
        use_rot = rng.random(n_pop) < rot_prob
    else:
        use_rot = np.zeros(n_pop, dtype=bool)
        
    # Cartesian crossover for non-rotated individuals
    idx_cart = np.where(~use_rot)[0]
    if len(idx_cart) > 0:
        trial[idx_cart] = binomial_crossover(target[idx_cart], donor[idx_cart], Cr[idx_cart], rng=rng)
        
    # Eigen crossover for rotated individuals
    idx_rot = np.where(use_rot)[0]
    if len(idx_rot) > 0:
        target_sub = target[idx_rot]
        donor_sub = donor[idx_rot]
        target_rot = project_to_eigen_basis(target_sub, eigen_basis)
        donor_rot = project_to_eigen_basis(donor_sub, eigen_basis)
        
        trial_rot = binomial_crossover(target_rot, donor_rot, Cr[idx_rot], rng=rng)
        trial[idx_rot] = reproject_from_eigen_basis(trial_rot, eigen_basis)
        
    if return_used_mask:
        return trial, use_rot
    return trial


def riemannian_tangent_crossover(
    target: np.ndarray,
    donor: np.ndarray,
    Cr: np.ndarray,
    eigen_basis: np.ndarray,
    use_chart: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Riemannian Tangent Projection Crossover (Version 3.0).
    
    Decomposes the tangent displacement vector d = donor - target onto the
    orthonormal Riemannian principal tangent frame U = [u_1, ..., u_D]:
        alpha_i = (v_i - x_i) @ U
    Performs binomial selection of independent geodesic modes:
        u_i = x_i + U [ m_i * alpha_i ]
    For individuals in the canonical chart (use_chart == False):
        u_i = x_i + m_i * (v_i - x_i)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n_pop, dim = target.shape
    d = donor - target
    
    # Binary binomial selection mask
    mask = rng.random((n_pop, dim)) <= Cr[:, np.newaxis]
    j_rand = rng.integers(0, dim, size=n_pop)
    for i in range(n_pop):
        mask[i, j_rand[i]] = True
        
    trials = target.copy()
    
    # Canonical Euclidean chart
    idx_can = np.where(~use_chart)[0]
    if len(idx_can) > 0:
        trials[idx_can] = np.where(mask[idx_can], donor[idx_can], target[idx_can])
        
    # Riemannian principal tangent chart
    idx_rot = np.where(use_chart)[0]
    if len(idx_rot) > 0:
        alpha = d[idx_rot] @ eigen_basis
        trials[idx_rot] = target[idx_rot] + (mask[idx_rot] * alpha) @ eigen_basis.T
        
    return trials

