"""Zero-Evaluation Online Landscape Sensor for RMA-EA (Version 2.0).

Calculates topological landscape roughness, condition number, and diversity
passively from the evolving population and Riemannian covariance without
spending ANY function evaluation budget on probe points.
"""

from typing import Optional
import numpy as np


class PassiveLandscapeSensor:
    """Zero-Cost Online Landscape Sensor.
    
    Monitors:
    1. Condition number ratio: kappa(C) = lambda_max / lambda_min (ill-conditioned vs spherical)
    2. Population evolutionary progress / stagnation: success rate SR
    3. Population spectral dispersion entropy
    
    Produces:
    - rho_t: ruggedness index in [0, 1]
    - p_rot: probability of performing crossover in the Riemannian eigen-coordinate system
    """
    
    def __init__(
        self,
        dim: int,
        smoothing: float = 0.2,
        history_len: int = 20
    ):
        self.dim = dim
        self.smoothing = smoothing
        self.history_len = history_len
        
        self.current_ruggedness: float = 0.5
        self.current_rot_prob: float = 0.8
        self.history_sr: list = []
        
    def update(
        self,
        eig_vals: np.ndarray,
        success_rate: float,
        fes_ratio: float
    ) -> float:
        """Update topological landscape metrics using zero extra evaluations.
        
        Args:
            eig_vals: Eigenvalues of elite covariance matrix (sorted descending)
            success_rate: Fraction of individuals that produced fitness improvement
            fes_ratio: Current FES / MaxFES in [0, 1]
        """
        self.history_sr.append(success_rate)
        if len(self.history_sr) > self.history_len:
            self.history_sr.pop(0)
            
        mean_sr = np.mean(self.history_sr)
        
        # 1. Condition number: ratio of principal to minor eigenvalue
        max_val = max(eig_vals[0], 1e-12)
        min_val = max(eig_vals[-1], 1e-12)
        log_cond = np.log10(max_val / min_val)
        
        # High condition number (> 3) indicates narrow ill-conditioned valley
        # Higher condition number -> higher rotation-basis crossover probability
        norm_cond = float(np.clip(log_cond / 6.0, 0.0, 1.0))
        self.current_rot_prob = float(np.clip(0.4 + 0.6 * norm_cond, 0.4, 0.95))
        
        # 2. Ruggedness estimation:
        # High success rate in early stages = smooth landscape
        # Very low success rate with high variance = rugged/stagnant trap
        # Early fes_ratio = more exploration
        stagnation_factor = float(np.clip((0.25 - mean_sr) / 0.25, 0.0, 1.0))
        raw_ruggedness = 0.5 * (1.0 - fes_ratio) + 0.5 * stagnation_factor
        
        # Smooth with exponential moving average
        self.current_ruggedness = (
            (1.0 - self.smoothing) * self.current_ruggedness +
            self.smoothing * raw_ruggedness
        )
        self.current_ruggedness = float(np.clip(self.current_ruggedness, 0.05, 0.95))
        return self.current_ruggedness
        
    def get_exploration_probability(self) -> float:
        """Dynamic exploration probability."""
        return self.current_ruggedness
        
    def get_eigen_crossover_probability(self) -> float:
        """Probability of projecting into Riemannian eigen-coordinate system for crossover."""
        return self.current_rot_prob
