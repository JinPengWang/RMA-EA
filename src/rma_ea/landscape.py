"""Online Landscape Ruggedness Sensor (LRS) for RMA-EA.

Monitors fitness landscape multimodality and curvature online to dynamically
balance exploration along geodesics and exploitation within Riemannian ellipsoids.
"""

from typing import Callable, Optional
import numpy as np


class LandscapeRuggednessSensor:
    """Online Landscape Ruggedness Sensor (LRS).
    
    Dynamically senses fitness landscape complexity (ruggedness, multi-funnel
    dispersion, and curvature variations) to adaptively balance exploration
    vs exploitation without manual tuning.
    """
    
    def __init__(
        self,
        dim: int,
        history_len: int = 15,
        kappa: float = 1.5,
        theta_base: float = 0.5,
        smoothing: float = 0.3,
        p_min: float = 0.1,
        p_max: float = 0.9,
        n_probe_samples: int = 5
    ):
        self.dim = dim
        self.history_len = history_len
        self.kappa = kappa
        self.theta_base = theta_base
        self.smoothing = smoothing
        self.p_min = p_min
        self.p_max = p_max
        self.n_probe_samples = n_probe_samples
        
        self.current_ruggedness: float = 0.5
        self.history_ruggedness: list = [0.5]
        
    def sense_and_update(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        eig_vecs: np.ndarray,
        eig_vals: np.ndarray,
        eval_func: Callable[[np.ndarray], np.ndarray],
        rng: Optional[np.random.Generator] = None,
        eps: float = 1e-10
    ) -> float:
        """Sense directional variation along principal axes and update ruggedness index.
        
        Evaluates second-order finite difference variations (directional ruggedness)
        along principal Riemannian geodesic axes:
        Curvature roughness = |f(x + delta*u) - 2f(x) + f(x - delta*u)| / delta^2
        """
        if rng is None:
            rng = np.random.default_rng()
            
        n_pop, dim = pop.shape
        k = len(eig_vals)
        n_probes = min(self.n_probe_samples, n_pop)
        
        # Sample probe individuals
        probe_indices = rng.choice(n_pop, size=n_probes, replace=False)
        probe_pop = pop[probe_indices]
        probe_fit = fitness[probe_indices]
        
        # Probing directions: choose top principal directions
        n_dirs = min(k, 3)
        curvatures = []
        
        for p_idx in range(n_probes):
            x = probe_pop[p_idx]
            f_orig = probe_fit[p_idx]
            for d in range(n_dirs):
                u_d = eig_vecs[:, d]
                delta = 0.05 * np.sqrt(max(eig_vals[d], 1e-3))
                
                x_plus = x + delta * u_d
                x_minus = x - delta * u_d
                
                f_plus = float(eval_func(x_plus.reshape(1, -1))[0])
                f_minus = float(eval_func(x_minus.reshape(1, -1))[0])
                
                # Second-order directional variation (ruggedness/non-quadratic roughness)
                second_diff = abs(f_plus - 2.0 * f_orig + f_minus) / (delta**2 + eps)
                curvatures.append(second_diff)
                
        curvatures = np.array(curvatures)
        if len(curvatures) > 1:
            mean_c = np.mean(curvatures)
            std_c = np.std(curvatures)
            # Coefficient of variation of directional curvature
            cv_c = std_c / (mean_c + eps)
        else:
            cv_c = 0.5
            
        # Sigmoidal mapping of CV to [0, 1]
        # In a quadratic function (Sphere), cv_c is close to 0 (constant curvature).
        # In multimodal functions (Rastrigin, etc.), cv_c is substantially larger.
        raw_ruggedness = 1.0 / (1.0 + np.exp(-self.kappa * (cv_c - self.theta_base)))
        
        # Smooth with exponential moving average
        self.current_ruggedness = (
            (1.0 - self.smoothing) * self.current_ruggedness + 
            self.smoothing * raw_ruggedness
        )
        self.current_ruggedness = float(np.clip(self.current_ruggedness, 0.0, 1.0))
        self.history_ruggedness.append(self.current_ruggedness)
        if len(self.history_ruggedness) > self.history_len:
            self.history_ruggedness.pop(0)
            
        return self.current_ruggedness
        
    def get_exploration_probability(self) -> float:
        """Calculate dynamic probability of executing geodesic exploration vs exploitation."""
        return float(self.p_min + (self.p_max - self.p_min) * self.current_ruggedness)
