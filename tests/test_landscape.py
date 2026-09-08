import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.landscape import LandscapeRuggednessSensor


def sphere_func(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=-1)


def rastrigin_func(x: np.ndarray) -> np.ndarray:
    return 10.0 * x.shape[-1] + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1)


def test_landscape_sensor_initialization():
    dim = 10
    sensor = LandscapeRuggednessSensor(dim=dim)
    assert 0.0 <= sensor.current_ruggedness <= 1.0
    p = sensor.get_exploration_probability()
    assert 0.0 <= p <= 1.0


def test_landscape_sensor_sphere_vs_rastrigin():
    dim = 10
    n_pop = 30
    rng = np.random.default_rng(42)
    
    # Random population in [-5, 5]
    pop = rng.uniform(-5.0, 5.0, (n_pop, dim))
    
    # Orthonormal eigenvectors
    eig_vecs, _ = np.linalg.qr(rng.standard_normal((dim, 4)))
    eig_vals = np.array([10.0, 5.0, 2.0, 1.0])
    
    sensor_sphere = LandscapeRuggednessSensor(dim=dim)
    fit_sphere = sphere_func(pop)
    for _ in range(5):
        r_sphere = sensor_sphere.sense_and_update(
            pop, fit_sphere, eig_vecs, eig_vals, sphere_func, rng=rng
        )
        
    sensor_rastrigin = LandscapeRuggednessSensor(dim=dim)
    fit_rastrigin = rastrigin_func(pop)
    for _ in range(5):
        r_rastrigin = sensor_rastrigin.sense_and_update(
            pop, fit_rastrigin, eig_vecs, eig_vals, rastrigin_func, rng=rng
        )
        
    # Rastrigin must exhibit higher ruggedness index than Sphere
    assert sensor_rastrigin.current_ruggedness > sensor_sphere.current_ruggedness


def test_exploration_probability_bounds():
    dim = 5
    sensor = LandscapeRuggednessSensor(dim=dim, p_min=0.15, p_max=0.85)
    
    sensor.current_ruggedness = 0.0
    assert np.isclose(sensor.get_exploration_probability(), 0.15)
    
    sensor.current_ruggedness = 1.0
    assert np.isclose(sensor.get_exploration_probability(), 0.85)
