import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from benchmarks.base import BenchmarkFunction
from benchmarks.cec_suite import get_benchmark_suite, CECBenchmark


def test_base_benchmark_shift_and_rotation():
    dim = 5
    shift = np.ones(dim) * 2.0
    # Identity rotation
    rot = np.eye(dim)
    
    # Simple sphere function
    def raw_sphere(z):
        return np.sum(z**2, axis=-1)
        
    bench = BenchmarkFunction(
        name="TestSphere",
        dim=dim,
        raw_func=raw_sphere,
        optimum_x=shift,
        bias=100.0,
        shift_vector=shift,
        rotation_matrix=rot,
        bounds=(-100.0, 100.0)
    )
    
    # At optimum_x, value should equal bias (100.0)
    val_opt = bench(shift.reshape(1, -1))
    assert np.isclose(val_opt[0], 100.0)


def test_cec_suite_functions():
    suite = get_benchmark_suite(dim=10)
    assert len(suite) >= 10
    
    # Test each function evaluates correctly on batch of points
    rng = np.random.default_rng(42)
    pop = rng.uniform(-100.0, 100.0, (15, 10))
    
    for func in suite:
        f_vals = func(pop)
        assert len(f_vals) == 15
        assert np.all(np.isfinite(f_vals))
        opt_val = func(func.optimum_x.reshape(1, -1))[0]
        if func.category == "Composition":
            assert abs(opt_val - func.bias) < 5.0
        else:
            assert np.isclose(opt_val, func.bias, atol=1e-5)
