import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analysis.statistics import (
    wilcoxon_test,
    friedman_test,
    holm_posthoc,
    summarize_run_statistics
)


def test_summarize_run_statistics():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    summary = summarize_run_statistics(data)
    assert np.isclose(summary["mean"], 3.0)
    assert np.isclose(summary["median"], 3.0)
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0


def test_wilcoxon_test_distinction():
    # Sample A clearly superior to Sample B
    sample_a = np.array([0.01, 0.02, 0.015, 0.009, 0.021, 0.018, 0.012, 0.019, 0.011, 0.014])
    sample_b = np.array([5.0, 4.8, 5.2, 5.1, 4.9, 5.3, 5.0, 5.2, 4.7, 5.1])
    
    verdict, stat, p_val = wilcoxon_test(sample_a, sample_b, alpha=0.05)
    assert verdict == "+"
    assert p_val < 0.01
    
    # Equal samples
    verdict_eq, _, p_eq = wilcoxon_test(sample_a, sample_a, alpha=0.05)
    assert verdict_eq == "~"


def test_friedman_and_holm():
    # 5 problems, 3 algorithms
    # Algo 0 is consistently best
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [1.2, 2.2, 3.2],
        [0.8, 1.8, 2.8],
        [1.1, 2.1, 3.1]
    ])
    algo_names = ["RMA-EA", "L-SHADE", "StandardDE"]
    res = friedman_test(matrix, algo_names)
    
    assert res["average_ranks"]["RMA-EA"] == 1.0
    assert res["average_ranks"]["L-SHADE"] == 2.0
    assert res["average_ranks"]["StandardDE"] == 3.0
    assert res["p_value"] < 0.01
    
    comparisons = holm_posthoc(res, control_name="RMA-EA", n_problems=5)
    assert len(comparisons) == 2
    assert comparisons[0]["competitor"] == "StandardDE"
