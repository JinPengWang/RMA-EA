import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.landscape import PassiveLandscapeSensor


def test_passive_landscape_sensor_initialization():
    dim = 10
    sensor = PassiveLandscapeSensor(dim=dim)
    assert 0.0 <= sensor.current_ruggedness <= 1.0
    assert 0.0 <= sensor.get_exploration_probability() <= 1.0
    assert 0.0 <= sensor.get_eigen_crossover_probability() <= 1.0


def test_passive_sensor_ill_conditioned_vs_spherical():
    dim = 10
    sensor = PassiveLandscapeSensor(dim=dim)
    
    # Spherical: all eigenvalues equal -> low condition number
    eig_spherical = np.ones(dim)
    sensor.update(eig_spherical, success_rate=0.4, fes_ratio=0.1)
    rot_prob_spherical = sensor.get_eigen_crossover_probability()
    
    # Ill-conditioned: ratio 10^6
    eig_ill = np.ones(dim)
    eig_ill[0] = 1e6
    sensor.update(eig_ill, success_rate=0.1, fes_ratio=0.1)
    rot_prob_ill = sensor.get_eigen_crossover_probability()
    
    # Ill-conditioned landscape must trigger higher eigen-coordinate crossover probability
    assert rot_prob_ill > rot_prob_spherical


def test_passive_sensor_stagnation():
    dim = 5
    sensor = PassiveLandscapeSensor(dim=dim)
    
    # High success rate (smooth early progress)
    sensor.update(np.ones(dim), success_rate=0.5, fes_ratio=0.8)
    rug_smooth = sensor.get_exploration_probability()
    
    # Low success rate (stagnant/rugged trap)
    for _ in range(10):
        sensor.update(np.ones(dim), success_rate=0.0, fes_ratio=0.1)
    rug_stagnant = sensor.get_exploration_probability()
    
    assert rug_stagnant > rug_smooth
