import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.operators import (
    ParameterMemory,
    DualChannelMutation,
    binomial_crossover,
    repair_bounds
)


def test_parameter_memory_sampling_and_update():
    memory = ParameterMemory(memory_size=10)
    F, Cr = memory.sample_parameters(size=50)
    
    assert len(F) == 50
    assert len(Cr) == 50
    assert np.all(F > 0.0) and np.all(F <= 1.0)
    assert np.all(Cr >= 0.0) and np.all(Cr <= 1.0)
    
    # Simulate successful steps
    succ_F = np.array([0.5, 0.7, 0.6])
    succ_Cr = np.array([0.8, 0.9, 0.85])
    improvements = np.array([10.0, 50.0, 20.0])
    
    initial_M_F = memory.M_F.copy()
    memory.update_memory(succ_F, succ_Cr, improvements)
    
    # Check memory slot updated
    assert not np.array_equal(initial_M_F, memory.M_F)


def test_binomial_crossover_shape_and_j_rand():
    rng = np.random.default_rng(42)
    dim = 10
    n = 20
    target = np.zeros((n, dim))
    donor = np.ones((n, dim))
    Cr = np.zeros(n)  # Even with Cr=0, at least 1 dimension must be from donor
    
    trial = binomial_crossover(target, donor, Cr, rng=rng)
    assert trial.shape == (n, dim)
    # At least one component per individual must be from donor (1.0)
    assert np.all(np.sum(trial == 1.0, axis=1) >= 1)


def test_repair_bounds():
    dim = 5
    lower = -5.0 * np.ones(dim)
    upper = 5.0 * np.ones(dim)
    
    x = np.array([
        [-6.0, 0.0, 7.0, -10.0, 5.5],
        [0.0, 1.0, -2.0, 3.0, 4.0]
    ])
    target = np.zeros_like(x)
    
    x_repaired = repair_bounds(x, target, lower, upper)
    assert np.all(x_repaired >= lower)
    assert np.all(x_repaired <= upper)


def test_dual_channel_mutation_outputs():
    dim = 8
    n_pop = 25
    rng = np.random.default_rng(123)
    
    pop = rng.uniform(-5.0, 5.0, (n_pop, dim))
    fitness = np.sum(pop**2, axis=-1)
    pbest_idx = np.argsort(fitness)[:5]
    archive = rng.uniform(-5.0, 5.0, (10, dim))
    
    eig_vecs, _ = np.linalg.qr(rng.standard_normal((dim, 3)))
    eig_vals = np.array([5.0, 2.0, 1.0])
    sigma_res = 0.1
    
    mutator = DualChannelMutation(dim=dim)
    F = np.full(n_pop, 0.5)
    
    # Test Exploration Channel
    donors_expl = mutator.mutate(
        pop=pop,
        fitness=fitness,
        pbest_indices=pbest_idx,
        archive=archive,
        F=F,
        channel_prob=1.0,  # 100% exploration
        eig_vecs=eig_vecs,
        eig_vals=eig_vals,
        sigma_res=sigma_res,
        rng=rng
    )
    assert donors_expl.shape == (n_pop, dim)
    
    # Test Exploitation Channel
    donors_expt = mutator.mutate(
        pop=pop,
        fitness=fitness,
        pbest_indices=pbest_idx,
        archive=archive,
        F=F,
        channel_prob=0.0,  # 100% exploitation
        eig_vecs=eig_vecs,
        eig_vals=eig_vals,
        sigma_res=sigma_res,
        rng=rng
    )
    assert donors_expt.shape == (n_pop, dim)
