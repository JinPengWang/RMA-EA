"""RMA-EA Core Algorithm Engine (Version 2.0 - High Performance SOTA Edition).

Integrates:
1. Log-Euclidean SPD Riemannian Manifold Elite Covariance Modeling.
2. Passive Zero-Cost Online Landscape Sensor (no FES wasted on probes).
3. Rotation-Invariant Riemannian Eigen-Coordinate Crossover.
4. Linear Population Size Reduction with Discarded-to-Archive Transfer.
5. Asymptotic Geodesic Perturbation Decay.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
import numpy as np

from .manifold import low_rank_covariance_decompose, symmetrize
from .landscape import PassiveLandscapeSensor
from .operators import (
    ParameterMemory,
    DualChannelMutation,
    riemannian_eigen_crossover,
    repair_bounds
)


@dataclass
class OptimizationResult:
    """Stores execution outcomes and convergence trajectories of RMA-EA."""
    best_x: np.ndarray
    best_f: float
    fes: int
    history_fitness: List[float] = field(default_factory=list)
    history_fes: List[int] = field(default_factory=list)
    history_ruggedness: List[float] = field(default_factory=list)
    history_pop_size: List[int] = field(default_factory=list)


class RMA_EA:
    """RMA-EA Optimizer (Version 2.0).
    
    Parameters:
        objective_func: Objective function accepting (N, D) array and returning (N,) array.
        dim: Problem dimensionality.
        lower_bound: Lower bound (scalar or array of shape (dim,)).
        upper_bound: Upper bound (scalar or array of shape (dim,)).
        max_fes: Maximum allowed function evaluations.
        pop_init: Initial population size (default: 18 * dim, min 50).
        pop_min: Minimum population size under LPSR (default: 4).
        p_best_rate: Fraction of top individuals considered as p-best (default: 0.11).
        arc_rate: Ratio of external archive size relative to current population (default: 1.4).
        memory_size: Size of historical parameter memory H (default: 20).
        manifold_active: Flag for ablation study (True = use Riemannian manifold).
        landscape_active: Flag for ablation study (True = use LRS sensing).
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        lower_bound: Union[float, np.ndarray],
        upper_bound: Union[float, np.ndarray],
        max_fes: int,
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
        self.max_fes = max_fes
        
        self.pop_init = pop_init if pop_init is not None else max(50, 18 * dim)
        self.pop_min = pop_min
        self.p_best_rate = p_best_rate
        self.arc_rate = arc_rate
        self.memory_size = memory_size
        
        if rank_k is None:
            self.rank_k = max(2, min(dim, int(np.ceil(np.sqrt(dim) * 1.5))))
        else:
            self.rank_k = min(dim, rank_k)
            
        self.manifold_active = manifold_active
        self.landscape_active = landscape_active
        self.rng = np.random.default_rng(seed)
        
        # Modules
        self.memory = ParameterMemory(memory_size=self.memory_size)
        self.mutator = DualChannelMutation(dim=self.dim)
        self.sensor = PassiveLandscapeSensor(dim=self.dim)
        
        # Adaptive operator selection and local intensification
        self.p_eigen = 0.3
        self.aos_alpha = 0.15
        self.rmli_active = True
        
    def optimize(self) -> OptimizationResult:
        """Run full RMA-EA optimization loop until max_fes is reached."""
        fes = 0
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
        history_fes = [fes]
        history_ruggedness = [self.sensor.current_ruggedness]
        history_pop_size = [current_pop_size]
        
        prev_success_rate = 0.2
        generation = 0
        
        # Generation loop
        while fes < self.max_fes:
            generation += 1
            # Sort population by fitness
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            if fitness[0] < global_best_f:
                global_best_f = float(fitness[0])
                global_best_x = pop[0].copy()
                
            # Elite individuals for covariance decomposition
            n_cov = max(min(current_pop_size, self.dim + 2), int(0.3 * current_pop_size))
            cov_indices = np.arange(n_cov)
            elite_samples = pop[cov_indices]
            
            fes_ratio = fes / float(self.max_fes)
            decay_factor = float((1.0 - fes_ratio)**2)
            
            # 1. Riemannian Manifold Covariance Decomposition
            if self.manifold_active:
                weights = np.log(n_cov + 0.5) - np.log(np.arange(1, n_cov + 1))
                weights /= np.sum(weights)
                
                # Truncated low-rank for perturbations
                eig_vals_k, eig_vecs_k, sigma_res = low_rank_covariance_decompose(
                    elite_samples, weights, rank_k=self.rank_k
                )
                
                # Full eigen-basis B for rotational invariance crossover
                mean_elite = np.sum(elite_samples * weights[:, np.newaxis], axis=0)
                diff_elite = elite_samples - mean_elite
                cov_full = (diff_elite.T * weights) @ diff_elite + np.eye(self.dim) * 1e-12
                all_vals, all_vecs = np.linalg.eigh(symmetrize(cov_full))
                sort_e = np.argsort(all_vals)[::-1]
                eigen_basis = all_vecs[:, sort_e]
                full_vals = np.maximum(all_vals[sort_e], 1e-12)
            else:
                eig_vecs_k = np.eye(self.dim, self.rank_k)
                eig_vals_k = np.ones(self.rank_k)
                sigma_res = 0.5
                eigen_basis = np.eye(self.dim)
                full_vals = np.ones(self.dim)
                
            # 2. Passive Zero-Cost Landscape Sensing
            if self.landscape_active:
                channel_prob = self.sensor.update(
                    eig_vals=full_vals,
                    success_rate=prev_success_rate,
                    fes_ratio=fes_ratio
                )
            else:
                channel_prob = 0.5
                
            # 3. Sample adaptive parameters F and Cr
            F, Cr = self.memory.sample_parameters(size=current_pop_size, rng=self.rng)
            
            # 4. pbest selection
            pbest_num = max(2, int(np.ceil(self.p_best_rate * current_pop_size)))
            pbest_indices = [self.rng.integers(0, pbest_num) for _ in range(current_pop_size)]
            
            donors = self.mutator.mutate(
                pop=pop,
                fitness=fitness,
                pbest_indices=pbest_indices,
                archive=archive,
                F=F,
                channel_prob=channel_prob,
                eig_vecs=eig_vecs_k,
                eig_vals=eig_vals_k,
                sigma_res=sigma_res,
                decay_factor=decay_factor,
                rng=self.rng
            )
            
            # 5. Success-History Adaptive Operator Selection (SH-AOS) Crossover with Dimension & Condition Gating
            if self.manifold_active and self.dim <= 10:
                log_cond = np.log10(full_vals[0] / full_vals[-1])
                rot_prob = 0.0 if log_cond > 3.5 else self.p_eigen
            else:
                rot_prob = 0.0
                
            trials, used_eigen = riemannian_eigen_crossover(
                target=pop,
                donor=donors,
                Cr=Cr,
                eigen_basis=eigen_basis if self.manifold_active else None,
                rot_prob=rot_prob,
                return_used_mask=True,
                rng=self.rng
            )
                
            # 6. Midpoint Bound Repair
            trials = repair_bounds(trials, pop, self.lower, self.upper)
            
            # 7. Evaluate Trial Vectors
            eval_size = min(len(trials), self.max_fes - fes)
            if eval_size < len(trials):
                trials = trials[:eval_size]
                pop_eval = pop[:eval_size]
                fitness_eval = fitness[:eval_size]
                F = F[:eval_size]
                Cr = Cr[:eval_size]
                used_eigen_eval = used_eigen[:eval_size]
            else:
                pop_eval = pop
                fitness_eval = fitness
                used_eigen_eval = used_eigen
                
            trial_fitness = self.func(trials)
            fes += eval_size
            
            # 8. Selection & Success Tracking
            improved_mask = trial_fitness < fitness_eval
            equal_mask = trial_fitness == fitness_eval
            accept_mask = improved_mask | equal_mask
            
            prev_success_rate = float(np.mean(improved_mask))
            
            successful_F = F[improved_mask]
            successful_Cr = Cr[improved_mask]
            fitness_improvements = fitness_eval[improved_mask] - trial_fitness[improved_mask]
            
            # Update archive with superseded parents
            if np.any(improved_mask):
                superseded_parents = pop_eval[improved_mask]
                archive = np.vstack([archive, superseded_parents])
                max_archive_size = int(self.arc_rate * current_pop_size)
                if len(archive) > max_archive_size:
                    rand_perm = self.rng.permutation(len(archive))[:max_archive_size]
                    archive = archive[rand_perm]
                    
                # Update SH-AOS probability if eigen-crossover was active
                if rot_prob > 0.0:
                    diff_f = fitness_eval[improved_mask] - trial_fitness[improved_mask]
                    imp_eigen = np.sum(diff_f[used_eigen_eval[improved_mask]]) if np.any(used_eigen_eval[improved_mask]) else 0.0
                    imp_cart = np.sum(diff_f[~used_eigen_eval[improved_mask]]) if np.any(~used_eigen_eval[improved_mask]) else 0.0
                    total_imp = imp_eigen + imp_cart
                    if total_imp > 0:
                        target_p = imp_eigen / total_imp
                        self.p_eigen = float(np.clip((1.0 - self.aos_alpha) * self.p_eigen + self.aos_alpha * target_p, 0.05, 0.95))
                    
            pop_eval[accept_mask] = trials[accept_mask]
            fitness_eval[accept_mask] = trial_fitness[accept_mask]
            
            # 9. Update Parameter Memory
            self.memory.update_memory(successful_F, successful_Cr, fitness_improvements)
            
            # 10. Linear Population Size Reduction (LPSR)
            target_pop_size = int(np.round(
                ((self.pop_min - self.pop_init) / float(self.max_fes)) * fes + self.pop_init
            ))
            target_pop_size = max(self.pop_min, target_pop_size)
            
            if target_pop_size < current_pop_size:
                sort_idx = np.argsort(fitness)
                # Transfer pruned individuals into archive for diversity preservation
                pruned_individuals = pop[sort_idx[target_pop_size:]]
                archive = np.vstack([archive, pruned_individuals])
                max_archive_size = int(self.arc_rate * target_pop_size)
                if len(archive) > max_archive_size:
                    rand_perm = self.rng.permutation(len(archive))[:max_archive_size]
                    archive = archive[rand_perm]
                    
                pop = pop[sort_idx[:target_pop_size]]
                fitness = fitness[sort_idx[:target_pop_size]]
                current_pop_size = target_pop_size
                
            # 11. Riemannian Manifold Local Intensification (RMLI)
            if self.rmli_active and self.manifold_active and fes_ratio > 0.85 and generation % 10 == 0 and fes + 2 <= self.max_fes:
                v_1 = eigen_basis[:, 0]
                delta_step = max(1e-5, float(np.sqrt(full_vals[0]) * 0.05))
                x_b = pop[0].copy()
                f_b = float(fitness[0])
                x_p = np.clip(x_b + delta_step * v_1, self.lower, self.upper)
                x_m = np.clip(x_b - delta_step * v_1, self.lower, self.upper)
                f_p = float(self.func(x_p[np.newaxis, :])[0])
                f_m = float(self.func(x_m[np.newaxis, :])[0])
                fes += 2
                denom = 2.0 * (f_p - 2.0 * f_b + f_m)
                if denom > 1e-14:
                    step_opt = -(f_p - f_m) * delta_step / denom
                    x_opt = np.clip(x_b + step_opt * v_1, self.lower, self.upper)
                    if fes < self.max_fes:
                        f_opt = float(self.func(x_opt[np.newaxis, :])[0])
                        fes += 1
                        if f_opt < f_b:
                            pop[0] = x_opt
                            fitness[0] = f_opt
                
            cur_best = float(np.min(fitness))
            if cur_best < global_best_f:
                global_best_f = cur_best
                global_best_x = pop[np.argmin(fitness)].copy()
                
            history_fitness.append(global_best_f)
            history_fes.append(fes)
            history_ruggedness.append(self.sensor.current_ruggedness)
            history_pop_size.append(current_pop_size)
            
        return OptimizationResult(
            best_x=global_best_x,
            best_f=global_best_f,
            fes=fes,
            history_fitness=history_fitness,
            history_fes=history_fes,
            history_ruggedness=history_ruggedness,
            history_pop_size=history_pop_size
        )
