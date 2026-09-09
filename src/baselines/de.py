"""Standard Differential Evolution (DE/rand/1/bin)."""

from typing import Callable, Union, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BaselineResult:
    best_x: np.ndarray
    best_f: float
    fes: int
    iterations: int = 0
    history_fitness: List[float] = field(default_factory=list)
    history_iterations: List[int] = field(default_factory=list)
    history_fes: List[int] = field(default_factory=list)


class StandardDE:
    """Standard DE/rand/1/bin baseline optimizer (Storn & Price, 1997)."""
    
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        lower_bound: Union[float, np.ndarray],
        upper_bound: Union[float, np.ndarray],
        max_iter: Optional[int] = 1000,
        max_fes: Optional[int] = None,
        pop_size: int = 100,
        F: float = 0.5,
        Cr: float = 0.9,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.dim = dim
        self.lower = np.full(dim, lower_bound, dtype=np.float64) if np.isscalar(lower_bound) else np.asarray(lower_bound, dtype=np.float64)
        self.upper = np.full(dim, upper_bound, dtype=np.float64) if np.isscalar(upper_bound) else np.asarray(upper_bound, dtype=np.float64)
        self.max_iter = max_iter
        self.max_fes = max_fes
        self.pop_size = pop_size
        self.F = F
        self.Cr = Cr
        self.rng = np.random.default_rng(seed)
        
    def optimize(self) -> BaselineResult:
        pop = self.rng.uniform(self.lower, self.upper, size=(self.pop_size, self.dim))
        fitness = self.func(pop)
        fes = len(pop)
        iteration = 0
        
        best_idx = np.argmin(fitness)
        best_f = float(fitness[best_idx])
        best_x = pop[best_idx].copy()
        
        hist_f = [best_f]
        hist_iter = [0]
        hist_fes = [fes]
        
        def should_terminate():
            if self.max_iter is not None and iteration >= self.max_iter:
                return True
            if self.max_fes is not None and fes >= self.max_fes:
                return True
            return False
            
        while not should_terminate():
            iteration += 1
            donors = np.zeros_like(pop)
            for i in range(self.pop_size):
                idxs = [j for j in range(self.pop_size) if j != i]
                r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
                donor = pop[r1] + self.F * (pop[r2] - pop[r3])
                donors[i] = donor
                
            # Binomial crossover
            rand_m = self.rng.random(pop.shape)
            mask = rand_m <= self.Cr
            j_rand = self.rng.integers(0, self.dim, size=self.pop_size)
            for i in range(self.pop_size):
                mask[i, j_rand[i]] = True
                
            trials = np.where(mask, donors, pop)
            # Bound handling: midpoint reflection
            mask_low = trials < self.lower
            mask_high = trials > self.upper
            trials[mask_low] = (pop[mask_low] + np.broadcast_to(self.lower, trials.shape)[mask_low]) / 2.0
            trials[mask_high] = (pop[mask_high] + np.broadcast_to(self.upper, trials.shape)[mask_high]) / 2.0
            trials = np.clip(trials, self.lower, self.upper)
            
            if self.max_fes is not None:
                eval_size = min(len(trials), self.max_fes - fes)
            else:
                eval_size = len(trials)
                
            if eval_size <= 0:
                break
                
            trials_eval = trials[:eval_size]
            pop_eval = pop[:eval_size]
            fitness_eval = fitness[:eval_size]
                
            trial_fit = self.func(trials_eval)
            fes += eval_size
            
            accept = trial_fit <= fitness_eval
            pop_eval[accept] = trials_eval[accept]
            fitness_eval[accept] = trial_fit[accept]
            
            cur_best = float(np.min(fitness))
            if cur_best < best_f:
                best_f = cur_best
                best_x = pop[np.argmin(fitness)].copy()
                
            hist_f.append(best_f)
            hist_iter.append(iteration)
            hist_fes.append(fes)
            
        return BaselineResult(
            best_x=best_x,
            best_f=best_f,
            fes=fes,
            iterations=iteration,
            history_fitness=hist_f,
            history_iterations=hist_iter,
            history_fes=hist_fes
        )
