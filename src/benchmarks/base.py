"""Base class for continuous numerical benchmark functions."""

from typing import Callable, Tuple, Optional
import numpy as np


class BenchmarkFunction:
    """Represents an objective function with coordinate shift, rotation, and bias."""
    
    def __init__(
        self,
        name: str,
        dim: int,
        raw_func: Callable[[np.ndarray], np.ndarray],
        optimum_x: Optional[np.ndarray] = None,
        bias: float = 0.0,
        shift_vector: Optional[np.ndarray] = None,
        rotation_matrix: Optional[np.ndarray] = None,
        bounds: Tuple[float, float] = (-100.0, 100.0),
        category: str = "Unimodal"
    ):
        self.name = name
        self.dim = dim
        self.raw_func = raw_func
        self.bias = bias
        self.bounds = bounds
        self.category = category
        
        self.shift = np.zeros(dim) if shift_vector is None else np.asarray(shift_vector, dtype=np.float64)
        self.rotation = np.eye(dim) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=np.float64)
        
        if optimum_x is None:
            self.optimum_x = self.shift.copy()
        else:
            self.optimum_x = np.asarray(optimum_x, dtype=np.float64)
            
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply shift and orthogonal rotation: z = (x - o) @ M."""
        diff = x - self.shift
        return diff @ self.rotation
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate function for a batch of points shape (N, D) or single point (D,)."""
        x_arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        z = self.transform(x_arr)
        raw_val = self.raw_func(z)
        return raw_val + self.bias
        
    def error(self, x: np.ndarray) -> np.ndarray:
        """Calculate optimization error relative to global optimum: f(x) - bias."""
        return np.maximum(0.0, self(x) - self.bias)
