import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baselines.de import StandardDE
from baselines.lshade import LSHADE
from baselines.cmaes import CMAES


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=-1)


def test_standard_de():
    dim = 5
    de = StandardDE(
        objective_func=sphere,
        dim=dim,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_fes=10000,
        pop_size=50,
        seed=42
    )
    res = de.optimize()
    assert res.best_f < 1e-4
    assert res.fes <= 10000


def test_lshade():
    dim = 5
    lshade = LSHADE(
        objective_func=sphere,
        dim=dim,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_fes=10000,
        pop_init=50,
        seed=42
    )
    res = lshade.optimize()
    assert res.best_f < 1e-5
    assert res.fes <= 10000


def test_cmaes():
    dim = 5
    cma = CMAES(
        objective_func=sphere,
        dim=dim,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_fes=10000,
        seed=42
    )
    res = cma.optimize()
    assert res.best_f < 1e-5
    assert res.fes <= 10000
