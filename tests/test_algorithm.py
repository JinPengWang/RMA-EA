import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA, OptimizationResult


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=-1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    # f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1.0)**2, axis=-1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    return 10.0 * x.shape[-1] + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def test_rma_ea_initialization():
    dim = 5
    optimizer = RMA_EA(
        objective_func=sphere,
        dim=dim,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_fes=1000,
        seed=42
    )
    assert optimizer.dim == dim
    assert optimizer.max_fes == 1000


def test_rma_ea_sphere_optimization():
    dim = 5
    optimizer = RMA_EA(
        objective_func=sphere,
        dim=dim,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_fes=10000,
        pop_init=50,
        seed=123
    )
    result = optimizer.optimize()
    
    assert isinstance(result, OptimizationResult)
    assert result.best_f < 1e-6  # Effectively solved to machine precision
    assert result.fes <= 10000
    assert len(result.history_fitness) > 0
    assert len(result.history_ruggedness) > 0


def test_rma_ea_rosenbrock_optimization():
    dim = 4
    optimizer = RMA_EA(
        objective_func=rosenbrock,
        dim=dim,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_fes=20000,
        pop_init=60,
        seed=42
    )
    result = optimizer.optimize()
    # Rosenbrock has global minimum at x*=(1,...,1), f*=0
    assert result.best_f < 1e-2


def test_rma_ea_rastrigin_multimodal():
    dim = 5
    optimizer = RMA_EA(
        objective_func=rastrigin,
        dim=dim,
        lower_bound=-5.12,
        upper_bound=5.12,
        max_fes=20000,
        pop_init=60,
        seed=777
    )
    result = optimizer.optimize()
    # Complex multimodal function should successfully locate near-zero basin
    assert result.best_f < 1e-1
