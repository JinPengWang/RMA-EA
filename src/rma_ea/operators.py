"""Evolutionary Operators and Adaptive Parameter Memory for RMA-EA.

Implements dual-channel geodesic mutation, binomial crossover, parameter memory (SHADE-style
Lehmer updates), and bound repair.
"""

from typing import Tuple, Optional
import numpy as np
from rma_ea.manifold import sample_geodesic_perturbation


def repair_bounds(
    x: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> np.ndarray:
    """Repair out-of-bound variables using midpoint projection to avoid edge accumulation.
    
    If x_j < lower_j: x_j = (target_j + lower_j) / 2
    If x_j > upper_j: x_j = (target_j + upper_j) / 2
    """
    repaired = x.copy()
    
    # Broadcast lower and upper if necessary
    if lower.ndim == 1 and repaired.ndim == 2:
        lower_b = lower[np.newaxis, :]
        upper_b = upper[np.newaxis, :]
    else:
        lower_b = lower
        upper_b = upper
        
    mask_low = repaired < lower_b
    mask_high = repaired > upper_b
    
    repaired[mask_low] = (target[mask_low] + np.broadcast_to(lower_b, repaired.shape)[mask_low]) / 2.0
    repaired[mask_high] = (target[mask_high] + np.broadcast_to(upper_b, repaired.shape)[mask_high]) / 2.0
    
    return np.clip(repaired, lower_b, upper_b)


class ParameterMemory:
    """Historical parameter memory for adaptive scale factor F and crossover rate Cr.
    
    Updates memory slots via successful Lehmer mean for F and weighted arithmetic mean for Cr.
    """
    
    def __init__(self, memory_size: int = 20, init_F: float = 0.5, init_Cr: float = 0.5):
        self.memory_size = memory_size
        self.M_F = np.full(memory_size, init_F, dtype=np.float64)
        self.M_Cr = np.full(memory_size, init_Cr, dtype=np.float64)
        self.memory_ptr = 0
        
    def sample_parameters(
        self,
        size: int,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample F from Cauchy and Cr from Gaussian around historical memory values."""
        if rng is None:
            rng = np.random.default_rng()
            
        r_indices = rng.integers(0, self.memory_size, size=size)
        
        # Sample F using Cauchy distribution
        sampled_F = np.zeros(size, dtype=np.float64)
        for i in range(size):
            while True:
                # Cauchy random variable: loc + scale * standard_cauchy
                f_val = self.M_F[r_indices[i]] + 0.1 * rng.standard_cauchy()
                if f_val > 1.0:
                    sampled_F[i] = 1.0
                    break
                elif f_val > 0.0:
                    sampled_F[i] = f_val
                    break
                    
        # Sample Cr using Gaussian distribution
        sampled_Cr = self.M_Cr[r_indices] + 0.1 * rng.standard_normal(size)
        sampled_Cr = np.clip(sampled_Cr, 0.0, 1.0)
        
        return sampled_F, sampled_Cr
        
    def update_memory(
        self,
        successful_F: np.ndarray,
        successful_Cr: np.ndarray,
        fitness_improvements: np.ndarray
    ) -> None:
        """Update memory slots using Lehmer mean for F and weighted arithmetic mean for Cr."""
        if len(successful_F) == 0:
            return
            
        total_imp = np.sum(fitness_improvements)
        if total_imp <= 0:
            weights = np.ones(len(fitness_improvements)) / len(fitness_improvements)
        else:
            weights = fitness_improvements / total_imp
            
        # Lehmer mean for F: sum(w * F^2) / sum(w * F)
        sum_w_f2 = np.sum(weights * (successful_F**2))
        sum_w_f = np.sum(weights * successful_F)
        mean_L_F = sum_w_f2 / max(sum_w_f, 1e-12)
        
        # Weighted arithmetic mean for Cr
        mean_A_Cr = np.sum(weights * successful_Cr)
        
        # Store in current memory slot and increment pointer
        self.M_F[self.memory_ptr] = float(np.clip(mean_L_F, 0.01, 1.0))
        self.M_Cr[self.memory_ptr] = float(np.clip(mean_A_Cr, 0.0, 1.0))
        
        self.memory_ptr = (self.memory_ptr + 1) % self.memory_size


class DualChannelMutation:
    """Dual-channel mutation operator arbitrating between exploration and exploitation.
    
    Channel A: Geodesic Drift Exploration (heavy-tailed manifold traversal)
    Channel B: Riemannian Anisotropic Contraction (tight refinement along curvature)
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def mutate(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        pbest_indices: np.ndarray,
        archive: np.ndarray,
        F: np.ndarray,
        channel_prob: float,
        eig_vecs: np.ndarray,
        eig_vals: np.ndarray,
        sigma_res: float,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Generate donor vectors for the entire population."""
        if rng is None:
            rng = np.random.default_rng()
            
        n_pop, dim = pop.shape
        donors = np.zeros_like(pop)
        
        # Candidate pool for r2: current population + external archive
        if archive is not None and len(archive) > 0:
            union_pool = np.vstack([pop, archive])
        else:
            union_pool = pop
        n_union = len(union_pool)
        
        # Sample perturbations for exploration and exploitation channels
        geodesic_perturbations = sample_geodesic_perturbation(
            eig_vecs, eig_vals, sigma_res, size=n_pop, rng=rng
        )
        
        for i in range(n_pop):
            # Select p-best individual
            pbest_idx = rng.choice(pbest_indices)
            x_pbest = pop[pbest_idx]
            
            # Select r1 from pop != i
            r1_choices = [idx for idx in range(n_pop) if idx != i]
            r1 = rng.choice(r1_choices)
            x_r1 = pop[r1]
            
            # Select r2 from union_pool != r1 and != i
            r2_choices = [idx for idx in range(n_union) if idx != i and idx != r1]
            if len(r2_choices) == 0:
                r2_choices = [idx for idx in range(n_union) if idx != i]
            r2 = rng.choice(r2_choices)
            x_r2 = union_pool[r2]
            
            f_i = F[i]
            
            # Dynamic Channel Decision: Exploration vs Exploitation
            if rng.random() < channel_prob:
                # Channel A: Geodesic Drift Exploration
                # current-to-pbest + difference vector + manifold geodesic perturbation
                cauchy_noise = 0.05 * f_i * rng.standard_cauchy(dim)
                cauchy_noise = np.clip(cauchy_noise, -1.0, 1.0)
                donor = (
                    pop[i] + 
                    f_i * (x_pbest - pop[i]) + 
                    f_i * (x_r1 - x_r2) + 
                    0.2 * f_i * geodesic_perturbations[i] + 
                    cauchy_noise
                )
            else:
                # Channel B: Riemannian Anisotropic Contraction Exploitation
                # pbest-centered contraction along principal covariance axes
                donor = (
                    x_pbest + 
                    f_i * (x_r1 - x_r2) + 
                    0.05 * f_i * geodesic_perturbations[i]
                )
                
            donors[i] = donor
            
        return donors


def binomial_crossover(
    target: np.ndarray,
    donor: np.ndarray,
    Cr: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Binomial crossover ensuring at least one component from donor."""
    if rng is None:
        rng = np.random.default_rng()
        
    n_pop, dim = target.shape
    trial = target.copy()
    
    rand_matrix = rng.random((n_pop, dim))
    mask = rand_matrix <= Cr[:, np.newaxis]
    
    # Ensure at least one dimension is inherited from donor
    j_rand = rng.integers(0, dim, size=n_pop)
    for i in range(n_pop):
        mask[i, j_rand[i]] = True
        
    trial[mask] = donor[mask]
    return trial
