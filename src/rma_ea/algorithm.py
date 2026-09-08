"""RMA-EA Core Algorithm Engine.

Riemannian Manifold & Landscape-Ruggedness Adaptive Evolutionary Algorithm.
Integrates SPD Log-Euclidean manifold guidance, online landscape ruggedness sensing,
dual-channel mutation, and Linear Population Size Reduction (LPSR).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union
import numpy as np

from rma_ea.manifold import low_rank_covariance_decompose
from rma_ea.landscape import LandscapeRuggednessSensor
from rma_ea.operators import (
    ParameterMemory,
    DualChannelMutation,
    binomial_crossover,
    repair_bounds
)


@dataclass
class OptimizationResult:
    """Stores the execution outcomes and convergence trajectories of RMA-EA."""
    best_x: np.ndarray
    best_f: float
    fes: int
    history_fitness: List[float] = field(default_factory=list)
    history_fes: List[int] = field(default_factory=list)
    history_ruggedness: List[float] = field(default_factory=list)
    history_pop_size: List[int] = field(default_factory=list)


class RMA_EA:
    """RMA-EA Optimizer.
    
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
        rank_k: Truncated rank for low-rank manifold decomposition.
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
        self.sensor = LandscapeRuggednessSensor(dim=self.dim)
        
    def _evaluate(self, pop: np.ndarray) -> np.ndarray:
        """Evaluate population and increment function evaluations counter."""
        fitness = self.func(pop)
        self.fes += len(pop)
        return fitness
        
    def optimize(self) -> OptimizationResult:
        """Run full RMA-EA optimization loop until max_fes is reached."""
        self.fes = 0
        current_pop_size = self.pop_init
        
        # Initialize population uniformly
        pop = self.rng.uniform(self.lower, self.upper, size=(current_pop_size, self.dim))
        fitness = self._evaluate(pop)
        
        # Initialize global best
        best_idx = np.argmin(fitness)
        global_best_f = float(fitness[best_idx])
        global_best_x = pop[best_idx].copy()
        
        # External archive
        archive: Optional[np.ndarray] = None
        
        # Convergence tracking
        history_fitness = [global_best_f]
        history_fes = [self.fes]
        history_ruggedness = [self.sensor.current_ruggedness]
        history_pop_size = [current_pop_size]
        
        # Generation loop
        while self.fes < self.max_fes:
            # Sort population by fitness
            sort_indices = np.argsort(fitness)
            pop = pop[sort_indices]
            fitness = fitness[sort_indices]
            
            # Update best
            if fitness[0] < global_best_f:
                global_best_f = float(fitness[0])
                global_best_x = pop[0].copy()
                
            # Elite individuals for manifold decomposition
            n_pbest = max(2, int(np.ceil(self.p_best_rate * current_pop_size)))
            pbest_indices = np.arange(n_pbest)
            
            # 1. Riemannian Manifold Covariance Decomposition
            if self.manifold_active:
                elite_samples = pop[pbest_indices]
                # Log-decay weights for elite individuals
                weights = np.log(n_pbest + 0.5) - np.log(np.arange(1, n_pbest + 1))
                weights /= np.sum(weights)
                
                eig_vals, eig_vecs, sigma_res = low_rank_covariance_decompose(
                    elite_samples, weights, rank_k=self.rank_k
                )
            else:
                # Euclidean fallback (identity covariance)
                eig_vecs = np.eye(self.dim, self.rank_k)
                eig_vals = np.ones(self.rank_k)
                sigma_res = 0.5
                
            # 2. Online Landscape Ruggedness Sensing
            if self.landscape_active and self.fes + 10 <= self.max_fes:
                # Sensor probe evaluation wrapped with FES accounting
                def probe_eval(x_arr: np.ndarray) -> np.ndarray:
                    return self._evaluate(x_arr)
                    
                channel_prob = self.sensor.sense_and_update(
                    pop, fitness, eig_vecs, eig_vals, probe_eval, rng=self.rng
                )
                channel_prob = self.sensor.get_exploration_probability()
            else:
                channel_prob = 0.5  # Neutral default when LRS is inactive
                
            # Check FES limit
            if self.fes >= self.max_fes:
                break
                
            # 3. Sample adaptive parameters F and Cr
            F, Cr = self.memory.sample_parameters(size=current_pop_size, rng=self.rng)
            
            # 4. Dual-Channel Mutation
            donors = self.mutator.mutate(
                pop=pop,
                fitness=fitness,
                pbest_indices=pbest_indices,
                archive=archive,
                F=F,
                channel_prob=channel_prob,
                eig_vecs=eig_vecs,
                eig_vals=eig_vals,
                sigma_res=sigma_res,
                rng=self.rng
            )
            
            # 5. Binomial Crossover
            trials = binomial_crossover(pop, donors, Cr, rng=self.rng)
            
            # 6. Bound Repair
            trials = repair_bounds(trials, pop, self.lower, self.upper)
            
            # 7. Evaluate Trial Vectors (budget check)
            eval_size = min(len(trials), self.max_fes - self.fes)
            if eval_size < len(trials):
                trials = trials[:eval_size]
                pop_eval = pop[:eval_size]
                fitness_eval = fitness[:eval_size]
                F = F[:eval_size]
                Cr = Cr[:eval_size]
            else:
                pop_eval = pop
                fitness_eval = fitness
                
            trial_fitness = self._evaluate(trials)
            
            # 8. Selection & Success Feedback
            improved_mask = trial_fitness < fitness_eval
            equal_mask = trial_fitness == fitness_eval
            accept_mask = improved_mask | equal_mask
            
            successful_F = F[improved_mask]
            successful_Cr = Cr[improved_mask]
            fitness_improvements = fitness_eval[improved_mask] - trial_fitness[improved_mask]
            
            # Add superseded parents to archive
            if np.any(improved_mask):
                superseded_parents = pop_eval[improved_mask]
                if archive is None or len(archive) == 0:
                    archive = superseded_parents.copy()
                else:
                    archive = np.vstack([archive, superseded_parents])
                    
                # Bound archive capacity
                max_archive_size = int(self.arc_rate * current_pop_size)
                if len(archive) > max_archive_size:
                    rand_perm = self.rng.permutation(len(archive))[:max_archive_size]
                    archive = archive[rand_perm]
                    
            # Update population
            pop_eval[accept_mask] = trials[accept_mask]
            fitness_eval[accept_mask] = trial_fitness[accept_mask]
            
            # 9. Update Parameter Memory
            self.memory.update_memory(successful_F, successful_Cr, fitness_improvements)
            
            # 10. Linear Population Size Reduction (LPSR)
            target_pop_size = int(np.round(
                ((self.pop_min - self.pop_init) / float(self.max_fes)) * self.fes + self.pop_init
            ))
            target_pop_size = max(self.pop_min, target_pop_size)
            
            if target_pop_size < current_pop_size:
                # Keep the best target_pop_size individuals
                sort_idx = np.argsort(fitness)
                pop = pop[sort_idx[:target_pop_size]]
                fitness = fitness[sort_idx[:target_pop_size]]
                current_pop_size = target_pop_size
                
            # Log generation metrics
            cur_best = float(np.min(fitness))
            if cur_best < global_best_f:
                global_best_f = cur_best
                global_best_x = pop[np.argmin(fitness)].copy()
                
            history_fitness.append(global_best_f)
            history_fes.append(self.fes)
            history_ruggedness.append(self.sensor.current_ruggedness)
            history_pop_size.append(current_pop_size)
            
        return OptimizationResult(
            best_x=global_best_x,
            best_f=global_best_f,
            fes=self.fes,
            history_fitness=history_fitness,
            history_fes=history_fes,
            history_ruggedness=history_ruggedness,
            history_pop_size=history_pop_size
        )
