import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.manifold import (
    matrix_log,
    matrix_exp,
    log_euclidean_distance,
    geodesic,
    update_riemannian_barycenter,
    low_rank_covariance_decompose,
    sample_geodesic_perturbation
)


def generate_random_spd(dim: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    spd = A @ A.T + np.eye(dim) * 0.1
    return spd


def test_matrix_log_and_exp_inverse():
    dim = 5
    C = generate_random_spd(dim)
    log_C = matrix_log(C)
    exp_log_C = matrix_exp(log_C)
    
    assert np.allclose(C, exp_log_C, atol=1e-8)
    assert np.allclose(log_C, log_C.T, atol=1e-8)


def test_log_euclidean_distance_properties():
    dim = 4
    C1 = generate_random_spd(dim, seed=1)
    C2 = generate_random_spd(dim, seed=2)
    
    # Distance to self is 0
    assert log_euclidean_distance(C1, C1) < 1e-10
    
    # Symmetry
    d12 = log_euclidean_distance(C1, C2)
    d21 = log_euclidean_distance(C2, C1)
    assert np.isclose(d12, d21)
    assert d12 > 0


def test_geodesic_endpoints_and_midpoint():
    dim = 4
    C1 = generate_random_spd(dim, seed=10)
    C2 = generate_random_spd(dim, seed=20)
    
    # Endpoints
    gamma_0 = geodesic(C1, C2, 0.0)
    gamma_1 = geodesic(C1, C2, 1.0)
    assert np.allclose(gamma_0, C1, atol=1e-8)
    assert np.allclose(gamma_1, C2, atol=1e-8)
    
    # Midpoint distance equality
    gamma_half = geodesic(C1, C2, 0.5)
    d1 = log_euclidean_distance(C1, gamma_half)
    d2 = log_euclidean_distance(gamma_half, C2)
    assert np.isclose(d1, d2, atol=1e-6)


def test_riemannian_barycenter_update():
    dim = 6
    C_prev = generate_random_spd(dim, seed=100)
    C_new = generate_random_spd(dim, seed=101)
    
    # When learning rate = 0, stays at C_prev
    C_updated_0 = update_riemannian_barycenter(C_prev, C_new, learning_rate=0.0)
    assert np.allclose(C_updated_0, C_prev, atol=1e-8)
    
    # When learning rate = 1, becomes C_new
    C_updated_1 = update_riemannian_barycenter(C_prev, C_new, learning_rate=1.0)
    assert np.allclose(C_updated_1, C_new, atol=1e-8)


def test_low_rank_decompose_and_sampling():
    dim = 10
    n_samples = 30
    rng = np.random.default_rng(2026)
    samples = rng.standard_normal((n_samples, dim))
    weights = np.ones(n_samples) / n_samples
    
    rank_k = 4
    eig_vals, eig_vecs, sigma_res = low_rank_covariance_decompose(samples, weights, rank_k)
    
    assert len(eig_vals) == rank_k
    assert eig_vecs.shape == (dim, rank_k)
    assert sigma_res >= 0
    assert np.all(eig_vals >= 0)
    
    # Test sampling perturbation
    perturbation = sample_geodesic_perturbation(eig_vecs, eig_vals, sigma_res, size=5, rng=rng)
    assert perturbation.shape == (5, dim)
