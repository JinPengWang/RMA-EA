"""RMA-EA Core Algorithm Engine (Version 3.0 - Riemannian Geodesic Flow Optimization).

A mathematically unified evolutionary continuous optimizer derived from first principles
of Riemannian Differential Geometry on the Symmetric Positive Definite (SPD) manifold S++^D.

Mathematical Foundations:
1. Continuous Riemannian Metric Flow on S++^D with Bayesian Robbins-Monro Warm-up:
   Maintains the cometric tensor C_t = G_t^{-1} without finite-sample rank deficiency:
       C_{t+1} = (1 - c_c(t)) C_t + c_c(t) * (C_emp / (Tr(C_emp)/D))
       c_c(t) = max(2.0 / (t + 2.0), 2.0 / D^{1.5})
2. Orthonormal Tangent Space Bundle:
   Spectral decomposition of the cometric tensor:
       C_t = U Lambda U^T = sum_{j=1}^D lambda_j u_j u_j^T
   spans the intrinsic principal geodesic axes of the fitness manifold.
3. Riemannian Tangent Projection Crossover:
   Decomposes the tangent displacement vector d = v_i - x_i onto the principal frame:
       u_i = x_i + U [ m_i * (U^T (v_i - x_i)) ]
   Eliminating distortion on ill-conditioned ravines while preserving rotational invariance.
4. Tri-Parameter Joint Historical Memory:
   Jointly adapts (F, Cr, Chart) via Lehmer and fitness-improvement weighted updating.
5. Standardized Iteration-Based Optimization with Geodesic LPSR.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
import numpy as np

from .manifold import RiemannianMetricFlow, symmetrize
from .landscape import PassiveLandscapeSensor
from .operators import (
    ParameterMemory,
    riemannian_tangent_crossover,
    repair_bounds
)


@dataclass
class OptimizationResult:
    """Stores execution outcomes and convergence trajectories of RMA-EA."""
    best_x: np.ndarray
    best_f: float
    fes: int
    iterations: int = 0
    history_fitness: List[float] = field(default_factory=list)
    history_iterations: List[int] = field(default_factory=list)
    history_fes: List[int] = field(default_factory=list)
    history_ruggedness: List[float] = field(default_factory=list)
    history_pop_size: List[int] = field(default_factory=list)


class RMA_EA:
    """RMA-EA Optimizer (Version 3.0 - Unified Riemannian Geodesic Flow).
    
    Parameters:
        objective_func: Objective function accepting (N, D) array and returning (N,) array.
        dim: Problem dimensionality.
        lower_bound: Lower bound (scalar or array of shape (dim,)).
        upper_bound: Upper bound (scalar or array of shape (dim,)).
        max_iter: Maximum allowed iterations / generations (default: 1000).
        max_fes: Maximum allowed function evaluations (optional alternative).
        pop_init: Initial population size (default: 18 * dim, min 50).
        pop_min: Minimum population size under LPSR (default: 4).
        p_best_rate: Fraction of top individuals considered as p-best (default: 0.11).
        arc_rate: Ratio of external archive size relative to current population (default: 1.4).
        memory_size: Size of historical parameter memory H (default: 20).
        manifold_active: Flag for ablation study (True = use Riemannian manifold).
        landscape_active: Flag for ablation study (retained for backward compatibility).
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        lower_bound: Union[float, np.ndarray],
        upper_bound: Union[float, np.ndarray],
        max_iter: Optional[int] = 1000,
        max_fes: Optional[int] = None,
        pop_init: Optional[int] = None,
        pop_min: int = 4,
        p_best_rate: float = 0.11,
        arc_rate: float = 1.4,
        memory_size: int = 20,
        rank_k: Optional[int] = None,
        manifold_active: bool = True,
        landscape_active: bool = True,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.dim = dim
        self.lower = np.full(dim, lower_bound, dtype=np.float64) if np.isscalar(lower_bound) else np.asarray(lower_bound, dtype=np.float64)
        self.upper = np.full(dim, upper_bound, dtype=np.float64) if np.isscalar(upper_bound) else np.asarray(upper_bound, dtype=np.float64)
        self.max_iter = max_iter
        self.max_fes = max_fes
        
        self.pop_init = pop_init if pop_init is not None else max(50, 18 * dim)
        self.pop_min = pop_min
        self.p_best_rate = p_best_rate
        self.arc_rate = arc_rate
        self.memory_size = memory_size
        self.manifold_active = manifold_active
        self.landscape_active = landscape_active
        self.rng = np.random.default_rng(seed)
        
        # Riemannian Metric Flow & Multi-Chart Memory Engine
        self.metric_flow = RiemannianMetricFlow(dim=self.dim)
        self.memory = ParameterMemory(memory_size=self.memory_size)
        self.sensor = PassiveLandscapeSensor(dim=self.dim)
        
    def optimize(self) -> OptimizationResult:
        """Run full RMA-EA optimization loop until max_iter (or max_fes) is reached."""
        fes = 0
        iteration = 0
        current_pop_size = self.pop_init
        
        # Initialize population uniformly
        pop = self.rng.uniform(self.lower, self.upper, size=(current_pop_size, self.dim))
        fitness = self.func(pop)
        fes += len(pop)
        
        # Initialize global best
        best_idx = np.argmin(fitness)
        global_best_f = float(fitness[best_idx])
        global_best_x = pop[best_idx].copy()
        
        # External archive
        archive = np.empty((0, self.dim))
        
        # Convergence tracking
        history_fitness = [global_best_f]
        history_iter = [0]
        history_fes = [fes]
        history_ruggedness = [self.sensor.current_ruggedness]
        history_pop_size = [current_pop_size]
        
        def should_terminate():
            if self.max_iter is not None and iteration >= self.max_iter:
                return True
            if self.max_fes is not None and fes >= self.max_fes:
                return True
            return False
            
        # Generation / Iteration loop
        while not should_terminate():
            iteration += 1
            
            # 1. Sort population by fitness
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            if fitness[0] < global_best_f:
                global_best_f = float(fitness[0])
                global_best_x = pop[0].copy()
                
            # 2. Continuous Riemannian Metric Flow Update on S++^D
            n_elite = max(4, int(0.2 * current_pop_size))
            elite_samples = pop[:n_elite]
            weights = np.log(n_elite + 0.5) - np.log(np.arange(1, n_elite + 1))
            weights /= np.sum(weights)
            
            if self.manifold_active:
                eigen_basis, eig_vals = self.metric_flow.update(elite_samples, weights)
            else:
                eigen_basis = np.eye(self.dim)
                eig_vals = np.ones(self.dim)
                
            # 3. Sample Parameters & Multi-Chart Indicators from Joint Memory
            F, Cr, use_chart, slot_indices = self.memory.sample_parameters(
                size=current_pop_size, rng=self.rng, return_chart=True
            )
            if not self.manifold_active:
                use_chart = np.zeros(current_pop_size, dtype=bool)
                
            # 4. Tangent Mutation (current-to-pbest/1 with archive)
            union_pool = np.vstack([pop, archive]) if len(archive) > 0 else pop
            n_union = len(union_pool)
            
            pbest_max = max(2, int(np.ceil(self.p_best_rate * current_pop_size)))
            pbest_indices = self.rng.integers(0, pbest_max, size=current_pop_size)
            
            donors = np.zeros_like(pop)
            for i in range(current_pop_size):
                x_pbest = pop[pbest_indices[i]]
                r1 = self.rng.integers(0, current_pop_size - 1)
                if r1 >= i:
                    r1 += 1
                x_r1 = pop[r1]
                
                if n_union > 2:
                    while True:
                        r2 = self.rng.integers(0, n_union)
                        if r2 != i and r2 != r1:
                            break
                elif n_union == 2:
                    r2 = 1 if (i == 0 or r1 == 0) else 0
                else:
                    r2 = 0
                x_r2 = union_pool[r2]
                
                donors[i] = pop[i] + F[i] * (x_pbest - pop[i]) + F[i] * (x_r1 - x_r2)
                
            # 5. Riemannian Tangent Projection Crossover
            trials = riemannian_tangent_crossover(
                target=pop,
                donor=donors,
                Cr=Cr,
                eigen_basis=eigen_basis,
                use_chart=use_chart,
                rng=self.rng
            )
            
            # 6. Bound Repair
            trials = repair_bounds(trials, pop, self.lower, self.upper)
            
            # 7. Evaluate Trial Vectors
            if self.max_fes is not None:
                eval_size = min(len(trials), self.max_fes - fes)
            else:
                eval_size = len(trials)
                
            if eval_size <= 0:
                break
                
            trials_eval = trials[:eval_size]
            pop_eval = pop[:eval_size]
            fitness_eval = fitness[:eval_size]
            F_eval = F[:eval_size]
            Cr_eval = Cr[:eval_size]
            use_chart_eval = use_chart[:eval_size]
            
            trial_fitness = self.func(trials_eval)
            fes += eval_size
            
            # 8. Selection & Success Tracking
            improved_mask = trial_fitness < fitness_eval
            equal_mask = trial_fitness == fitness_eval
            accept_mask = improved_mask | equal_mask
            
            # Update archive & Tri-Parameter Memory
            if np.any(improved_mask):
                archive = np.vstack([archive, pop_eval[improved_mask]])
                max_archive_size = int(self.arc_rate * current_pop_size)
                if len(archive) > max_archive_size:
                    rand_perm = self.rng.permutation(len(archive))[:max_archive_size]
                    archive = archive[rand_perm]
                    
                successful_F = F_eval[improved_mask]
                successful_Cr = Cr_eval[improved_mask]
                successful_chart = use_chart_eval[improved_mask]
                fitness_improvements = fitness_eval[improved_mask] - trial_fitness[improved_mask]
                
                self.memory.update_memory(
                    successful_F=successful_F,
                    successful_Cr=successful_Cr,
                    fitness_improvements=fitness_improvements,
                    successful_chart=successful_chart
                )
                
            pop_eval[accept_mask] = trials_eval[accept_mask]
            fitness_eval[accept_mask] = trial_fitness[accept_mask]
            
            # 9. Linear Population Size Reduction (LPSR)
            if self.max_iter is not None:
                progress = min(1.0, float(iteration) / (self.max_iter * 0.75))
            else:
                progress = min(1.0, float(fes) / self.max_fes)
                
            target_pop_size = int(np.round(((self.pop_min - self.pop_init) * progress + self.pop_init)))
            target_pop_size = max(self.pop_min, target_pop_size)
            
            if target_pop_size < current_pop_size:
                sort_idx = np.argsort(fitness)
                pruned_individuals = pop[sort_idx[target_pop_size:]]
                archive = np.vstack([archive, pruned_individuals])
                max_archive_size = int(self.arc_rate * target_pop_size)
                if len(archive) > max_archive_size:
                    rand_perm = self.rng.permutation(len(archive))[:max_archive_size]
                    archive = archive[rand_perm]
                    
                pop = pop[sort_idx[:target_pop_size]]
                fitness = fitness[sort_idx[:target_pop_size]]
                current_pop_size = target_pop_size
                
            cur_best = float(np.min(fitness))
            if cur_best < global_best_f:
                global_best_f = cur_best
                global_best_x = pop[np.argmin(fitness)].copy()
                
            history_fitness.append(global_best_f)
            history_iter.append(iteration)
            history_fes.append(fes)
            history_ruggedness.append(float(np.log10(max(eig_vals[0] / eig_vals[-1], 1.0))))
            history_pop_size.append(current_pop_size)
            
        return OptimizationResult(
            best_x=global_best_x,
            best_f=global_best_f,
            fes=fes,
            iterations=iteration,
            history_fitness=history_fitness,
            history_iterations=history_iter,
            history_fes=history_fes,
            history_ruggedness=history_ruggedness,
            history_pop_size=history_pop_size
        )
