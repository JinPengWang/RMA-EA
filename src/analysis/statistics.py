"""Rigorous Non-Parametric Statistical Testing Suite for Evolutionary Computation.

Implements Wilcoxon signed-rank test, Friedman ranking test, Holm post-hoc procedure,
and comprehensive summary statistics in strict accordance with IEEE TEVC standards.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy import stats


def summarize_run_statistics(errors: np.ndarray) -> Dict[str, float]:
    """Calculate mean, standard deviation, median, IQR, min, and max of run errors."""
    arr = np.asarray(errors, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def wilcoxon_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    alpha: float = 0.05
) -> Tuple[str, float, float]:
    """Perform two-sided Wilcoxon signed-rank test between sample_a and sample_b.
    
    Returns:
        verdict: '+' if sample_a is significantly better (lower error) than sample_b,
                 '-' if sample_a is significantly worse than sample_b,
                 '~' if there is no statistically significant difference (p >= alpha).
        stat: test statistic
        p_val: asymptotic or exact p-value
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    
    diff = a - b
    # Check if all differences are zero
    if np.all(np.isclose(diff, 0.0, atol=1e-12)):
        return "~", 0.0, 1.0
        
    try:
        stat, p_val = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except Exception:
        stat, p_val = 0.0, 1.0
        
    if p_val < alpha:
        # Statistically significant difference exists
        median_diff = np.median(diff)
        mean_diff = np.mean(diff)
        if median_diff < 0 or (np.isclose(median_diff, 0.0) and mean_diff < 0):
            verdict = "+"  # sample_a has lower error (better)
        else:
            verdict = "-"  # sample_a has higher error (worse)
    else:
        verdict = "~"      # Indistinguishable
        
    return verdict, float(stat), float(p_val)


def friedman_test(
    score_matrix: np.ndarray,
    algorithm_names: List[str]
) -> Dict[str, Any]:
    """Perform the Friedman test across multiple problems and algorithms.
    
    Args:
        score_matrix: (N_problems, N_algorithms) matrix of performance metrics (lower is better).
        algorithm_names: List of algorithm names.
        
    Returns:
        dict containing average_ranks, chi2_stat, p_value, and ranking ordering.
    """
    n_problems, n_algos = score_matrix.shape
    
    # Compute rank per row (problem), 1 for best (lowest)
    ranks = np.zeros_like(score_matrix)
    for i in range(n_problems):
        ranks[i] = stats.rankdata(score_matrix[i], method="average")
        
    avg_ranks = np.mean(ranks, axis=0)
    
    # Friedman statistic
    # chi2_F = [12 * N / (k * (k + 1))] * [sum(R_j^2) - k * (k + 1)^2 / 4]
    term1 = 12.0 * n_problems / (n_algos * (n_algos + 1.0))
    term2 = np.sum(avg_ranks**2) - (n_algos * (n_algos + 1.0)**2) / 4.0
    chi2_stat = term1 * term2
    
    p_val = float(stats.chi2.sf(chi2_stat, df=n_algos - 1))
    
    rank_dict = {algorithm_names[j]: float(avg_ranks[j]) for j in range(n_algos)}
    sorted_algorithms = sorted(rank_dict.items(), key=lambda item: item[1])
    
    return {
        "average_ranks": rank_dict,
        "ranking_order": sorted_algorithms,
        "chi2_stat": float(chi2_stat),
        "p_value": p_val,
        "raw_ranks": ranks
    }


def holm_posthoc(
    friedman_results: Dict[str, Any],
    control_name: str,
    n_problems: int,
    alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """Holm post-hoc procedure comparing control algorithm against all other algorithms."""
    avg_ranks = friedman_results["average_ranks"]
    control_rank = avg_ranks[control_name]
    n_algos = len(avg_ranks)
    
    # Standard error SE = sqrt(k * (k + 1) / (6 * N))
    se = np.sqrt(n_algos * (n_algos + 1.0) / (6.0 * n_problems))
    
    comparisons = []
    for name, rank in avg_ranks.items():
        if name == control_name:
            continue
        z = (rank - control_rank) / se
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        comparisons.append({
            "comparison": f"{control_name} vs {name}",
            "competitor": name,
            "rank_diff": float(rank - control_rank),
            "z_value": float(z),
            "p_value": float(p)
        })
        
    # Sort ascending by p-value
    comparisons.sort(key=lambda x: x["p_value"])
    
    # Apply Holm step-down correction
    m = len(comparisons)
    for i, item in enumerate(comparisons):
        target_alpha = alpha / (m - i)
        item["target_alpha"] = float(target_alpha)
        item["significant"] = bool(item["p_value"] < target_alpha)
        
    return comparisons


def compute_win_tie_loss(
    algorithm_errors: Dict[str, Dict[str, np.ndarray]],
    control_name: str,
    competitors: List[str],
    problem_names: List[str]
) -> Dict[str, Dict[str, int]]:
    """Compute Win / Tie / Loss summary for control algorithm against each competitor."""
    summary = {comp: {"win": 0, "tie": 0, "loss": 0} for comp in competitors}
    
    for prob in problem_names:
        ctrl_runs = algorithm_errors[control_name][prob]
        for comp in competitors:
            comp_runs = algorithm_errors[comp][prob]
            verdict, _, _ = wilcoxon_test(ctrl_runs, comp_runs)
            if verdict == "+":
                summary[comp]["win"] += 1
            elif verdict == "-":
                summary[comp]["loss"] += 1
            else:
                summary[comp]["tie"] += 1
                
    return summary
