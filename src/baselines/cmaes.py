"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

Standard pure Python implementation of the (mu/mu_w, lambda)-CMA-ES (Hansen, 2016).
"""

from typing import Callable, Union, Optional
import numpy as np
from .de import BaselineResult


class CMAES:
    """CMA-ES baseline optimizer."""
    
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], np.ndarray],
        dim: int,
        lower_bound: Union[float, np.ndarray],
        upper_bound: Union[float, np.ndarray],
        max_fes: int,
        sigma0: Optional[float] = None,
        seed: Optional[int] = None
    ):
        self.func = objective_func
        self.dim = dim
        self.lower = np.full(dim, lower_bound, dtype=np.float64) if np.isscalar(lower_bound) else np.asarray(lower_bound, dtype=np.float64)
        self.upper = np.full(dim, upper_bound, dtype=np.float64) if np.isscalar(upper_bound) else np.asarray(upper_bound, dtype=np.float64)
        self.max_fes = max_fes
        self.rng = np.random.default_rng(seed)
        
        # Strategy parameters
        self.xmean = (self.lower + self.upper) / 2.0
        self.sigma = sigma0 if sigma0 is not None else float(np.mean(self.upper - self.lower) * 0.3)
        
        # Population parameters
        self.lam = int(4 + np.floor(3 * np.log(dim)))
        self.mu = self.lam // 2
        
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = float(np.sum(self.weights)**2 / np.sum(self.weights**2))
        
        # Adaptation parameters
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        
        # Dynamic state
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.chiN = dim**0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
    def optimize(self) -> BaselineResult:
        fes = 0
        best_f = float("inf")
        best_x = self.xmean.copy()
        
        hist_f = []
        hist_fes = []
        
        while fes < self.max_fes:
            # Generate lambda offspring
            z = self.rng.standard_normal((self.lam, self.dim))
            # x = m + sigma * B * D * z
            y = z * self.D
            artmp = y @ self.B.T
            arx = self.xmean + self.sigma * artmp
            
            # Boundary handling: clip
            arx_clipped = np.clip(arx, self.lower, self.upper)
            
            # Budget check
            eval_size = min(self.lam, self.max_fes - fes)
            if eval_size < self.lam:
                arx_eval = arx_clipped[:eval_size]
            else:
                arx_eval = arx_clipped
                
            fitness = self.func(arx_eval)
            fes += eval_size
            
            if eval_size < self.lam:
                # Pad to maintain CMA update dimensions if at budget boundary
                pad_size = self.lam - eval_size
                fitness = np.concatenate([fitness, np.full(pad_size, float("inf"))])
                
            # Sort offspring
            sort_idx = np.argsort(fitness)
            cur_best_f = float(fitness[sort_idx[0]])
            if cur_best_f < best_f:
                best_f = cur_best_f
                best_x = arx_clipped[sort_idx[0]].copy()
                
            hist_f.append(best_f)
            hist_fes.append(fes)
            
            if fes >= self.max_fes:
                break
                
            # Selection and recombination
            xold = self.xmean.copy()
            self.xmean = np.sum(arx[sort_idx[:self.mu]] * self.weights[:, np.newaxis], axis=0)
            
            # Evolution paths
            # y_w = (xmean - xold) / sigma
            y_w = (self.xmean - xold) / (self.sigma + 1e-12)
            # inv_sqrt_C * y_w = B * D^-1 * B.T * y_w
            z_w = self.B @ ((self.B.T @ y_w) / self.D)
            
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
            denom_hsig = np.sqrt(max(1e-12, 1.0 - (1.0 - self.cs)**(2.0 * (fes / self.lam)))) * self.chiN
            hsig = 1.0 if (np.linalg.norm(self.ps) / denom_hsig) < (1.4 + 2.0 / (self.dim + 1)) else 0.0
            
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
            
            # Adapt covariance matrix C
            artmp_mu = (arx[sort_idx[:self.mu]] - xold) / (self.sigma + 1e-12)
            rank_mu = (artmp_mu.T * self.weights) @ artmp_mu
            
            self.C = (
                (1 - self.c1 - self.cmu) * self.C +
                self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) +
                self.cmu * rank_mu
            )
            
            # Adapt step-size sigma
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
            
            # Decomposition C = B * D^2 * B.T
            self.C = 0.5 * (self.C + self.C.T)
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-14)
            self.D = np.sqrt(eigvals)
            self.B = eigvecs
            
        return BaselineResult(
            best_x=best_x,
            best_f=best_f,
            fes=fes,
            history_fitness=hist_f,
            history_fes=hist_fes
        )
