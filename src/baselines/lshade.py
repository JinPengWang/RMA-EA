"""L-SHADE (Linear Population Size Reduction Success-History Adaptive Differential Evolution).

Faithful implementation of the standard L-SHADE algorithm (Tanabe & Fukunaga, CEC 2014).
"""

from typing import Callable, Union, Optional
import numpy as np
from .de import BaselineResult


class LSHADE:
    """L-SHADE optimizer baseline."""
    
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
        self.rng = np.random.default_rng(seed)
        
    def optimize(self) -> BaselineResult:
        current_pop_size = self.pop_init
        pop = self.rng.uniform(self.lower, self.upper, size=(current_pop_size, self.dim))
        fitness = self.func(pop)
        fes = len(pop)
        
        best_idx = np.argmin(fitness)
        best_f = float(fitness[best_idx])
        best_x = pop[best_idx].copy()
        
        hist_f = [best_f]
        hist_fes = [fes]
        
        # Memory
        M_F = np.full(self.memory_size, 0.5)
        M_Cr = np.full(self.memory_size, 0.5)
        k_mem = 0
        
        archive = np.empty((0, self.dim))
        
        while fes < self.max_fes:
            # Sort population
            sort_idx = np.argsort(fitness)
            pop = pop[sort_idx]
            fitness = fitness[sort_idx]
            
            pbest_num = max(2, int(np.ceil(self.p_best_rate * current_pop_size)))
            
            # Sample F and Cr
            r_idx = self.rng.integers(0, self.memory_size, size=current_pop_size)
            F = np.zeros(current_pop_size)
            for i in range(current_pop_size):
                while True:
                    val = M_F[r_idx[i]] + 0.1 * self.rng.standard_cauchy()
                    if val > 1.0:
                        F[i] = 1.0
                        break
                    elif val > 0.0:
                        F[i] = val
                        break
                        
            Cr = M_Cr[r_idx] + 0.1 * self.rng.standard_normal(current_pop_size)
            Cr = np.clip(Cr, 0.0, 1.0)
            
            # Mutation (current-to-pbest/1/bin with archive)
            union_pool = np.vstack([pop, archive]) if len(archive) > 0 else pop
            donors = np.zeros_like(pop)
            
            pbest_indices = self.rng.integers(0, pbest_num, size=current_pop_size)
            n_union = len(union_pool)
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
                
            # Crossover
            mask = self.rng.random((current_pop_size, self.dim)) <= Cr[:, np.newaxis]
            j_rand = self.rng.integers(0, self.dim, size=current_pop_size)
            for i in range(current_pop_size):
                mask[i, j_rand[i]] = True
            trials = np.where(mask, donors, pop)
            
            # Bound repair
            mask_low = trials < self.lower
            mask_high = trials > self.upper
            trials[mask_low] = (pop[mask_low] + np.broadcast_to(self.lower, trials.shape)[mask_low]) / 2.0
            trials[mask_high] = (pop[mask_high] + np.broadcast_to(self.upper, trials.shape)[mask_high]) / 2.0
            trials = np.clip(trials, self.lower, self.upper)
            
            eval_size = min(len(trials), self.max_fes - fes)
            if eval_size < len(trials):
                trials = trials[:eval_size]
                pop_eval = pop[:eval_size]
                fitness_eval = fitness[:eval_size]
                F = F[:eval_size]
                Cr = Cr[:eval_size]
            else:
                pop_eval = pop
                fitness_eval = fitness
                
            trial_fit = self.func(trials)
            fes += eval_size
            
            improved = trial_fit < fitness_eval
            equal = trial_fit == fitness_eval
            accept = improved | equal
            
            succ_F = F[improved]
            succ_Cr = Cr[improved]
            diff_f = fitness_eval[improved] - trial_fit[improved]
            
            # Update archive
            if np.any(improved):
                archive = np.vstack([archive, pop_eval[improved]])
                max_arc = int(self.arc_rate * current_pop_size)
                if len(archive) > max_arc:
                    perm = self.rng.permutation(len(archive))[:max_arc]
                    archive = archive[perm]
                    
            pop_eval[accept] = trials[accept]
            fitness_eval[accept] = trial_fit[accept]
            
            # Update memory
            if len(succ_F) > 0:
                weights = diff_f / np.sum(diff_f) if np.sum(diff_f) > 0 else np.ones(len(diff_f)) / len(diff_f)
                M_F[k_mem] = np.sum(weights * (succ_F**2)) / max(np.sum(weights * succ_F), 1e-12)
                M_Cr[k_mem] = np.sum(weights * succ_Cr)
                k_mem = (k_mem + 1) % self.memory_size
                
            # LPSR
            target_pop = int(np.round(
                ((self.pop_min - self.pop_init) / float(self.max_fes)) * fes + self.pop_init
            ))
            target_pop = max(self.pop_min, target_pop)
            if target_pop < current_pop_size:
                s_idx = np.argsort(fitness)
                pop = pop[s_idx[:target_pop]]
                fitness = fitness[s_idx[:target_pop]]
                current_pop_size = target_pop
                
            cur_best = float(np.min(fitness))
            if cur_best < best_f:
                best_f = cur_best
                best_x = pop[np.argmin(fitness)].copy()
                
            hist_f.append(best_f)
            hist_fes.append(fes)
            
        return BaselineResult(
            best_x=best_x,
            best_f=best_f,
            fes=fes,
            history_fitness=hist_f,
            history_fes=hist_fes
        )
