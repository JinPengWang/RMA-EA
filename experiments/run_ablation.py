"""Ablation Study Experiment Suite for RMA-EA.

Isolates and validates the individual contributions of:
1. Riemannian Manifold Covariance Modeling (SPD manifold with Log-Euclidean metric)
2. Online Landscape Ruggedness Sensor (LRS)
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rma_ea.algorithm import RMA_EA
from baselines.de import StandardDE
from benchmarks.cec_suite import get_benchmark_suite
from analysis.statistics import summarize_run_statistics, wilcoxon_test, friedman_test


def run_ablation_study(
    dim: int = 10,
    n_runs: int = 20,
    max_fes: int = 20000,
    output_dir: str = "experiments/results"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    suite = get_benchmark_suite(dim=dim)
    
    variants = [
        "RMA-EA (Full)",
        "RMA-EA w/o Manifold",
        "RMA-EA w/o LRS",
        "StandardDE"
    ]
    problem_names = [f.name for f in suite]
    
    all_errors: Dict[str, Dict[str, List[float]]] = {
        var: {prob: [] for prob in problem_names} for var in variants
    }
    
    print(f"============================================================")
    print(f"Starting RMA-EA Ablation Study: Dim={dim}, Runs={n_runs}, MaxFES={max_fes}")
    print(f"Variants: {', '.join(variants)}")
    print(f"============================================================")
    
    start_time = time.time()
    
    for prob_idx, func in enumerate(suite):
        p_name = func.name
        print(f"\nAblation on Problem [{prob_idx + 1}/{len(suite)}]: {p_name}")
        
        for run_idx in range(n_runs):
            seed = 50000 + prob_idx * 500 + run_idx
            
            # 1. Full RMA-EA
            opt_full = RMA_EA(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                manifold_active=True,
                landscape_active=True,
                seed=seed
            )
            res_full = opt_full.optimize()
            all_errors["RMA-EA (Full)"][p_name].append(max(0.0, float(res_full.best_f - func.bias)))
            
            # 2. RMA-EA w/o Manifold
            opt_no_m = RMA_EA(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                manifold_active=False,
                landscape_active=True,
                seed=seed
            )
            res_no_m = opt_no_m.optimize()
            all_errors["RMA-EA w/o Manifold"][p_name].append(max(0.0, float(res_no_m.best_f - func.bias)))
            
            # 3. RMA-EA w/o LRS
            opt_no_lrs = RMA_EA(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                manifold_active=True,
                landscape_active=False,
                seed=seed
            )
            res_no_lrs = opt_no_lrs.optimize()
            all_errors["RMA-EA w/o LRS"][p_name].append(max(0.0, float(res_no_lrs.best_f - func.bias)))
            
            # 4. StandardDE
            opt_de = StandardDE(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                seed=seed
            )
            res_de = opt_de.optimize()
            all_errors["StandardDE"][p_name].append(max(0.0, float(res_de.best_f - func.bias)))
            
        m_full = np.mean(all_errors["RMA-EA (Full)"][p_name])
        m_no_m = np.mean(all_errors["RMA-EA w/o Manifold"][p_name])
        m_no_l = np.mean(all_errors["RMA-EA w/o LRS"][p_name])
        m_de = np.mean(all_errors["StandardDE"][p_name])
        print(f"  Mean Errors -> Full: {m_full:.2e} | w/o Manifold: {m_no_m:.2e} | w/o LRS: {m_no_l:.2e} | DE: {m_de:.2e}")
        
    elapsed = time.time() - start_time
    print(f"\nAblation experiments completed in {elapsed:.2f} seconds.")
    
    mean_matrix = np.zeros((len(problem_names), len(variants)))
    summary_table: Dict[str, Dict[str, Dict[str, float]]] = {
        prob: {} for prob in problem_names
    }
    
    for p_idx, prob in enumerate(problem_names):
        for v_idx, var in enumerate(variants):
            arr = np.array(all_errors[var][prob])
            stats_dict = summarize_run_statistics(arr)
            summary_table[prob][var] = stats_dict
            mean_matrix[p_idx, v_idx] = stats_dict["mean"]
            
    friedman_res = friedman_test(mean_matrix, variants)
    
    # Wilcoxon against Full RMA-EA
    wilcoxon_comp: Dict[str, Dict[str, Any]] = {
        var: {} for var in ["RMA-EA w/o Manifold", "RMA-EA w/o LRS", "StandardDE"]
    }
    win_tie_loss: Dict[str, Dict[str, int]] = {
        var: {"+": 0, "~": 0, "-": 0} for var in ["RMA-EA w/o Manifold", "RMA-EA w/o LRS", "StandardDE"]
    }
    
    for prob in problem_names:
        full_runs = np.array(all_errors["RMA-EA (Full)"][prob])
        for var in ["RMA-EA w/o Manifold", "RMA-EA w/o LRS", "StandardDE"]:
            var_runs = np.array(all_errors[var][prob])
            verdict, stat, p_val = wilcoxon_test(full_runs, var_runs)
            wilcoxon_comp[var][prob] = {"verdict": verdict, "stat": stat, "p_val": p_val}
            win_tie_loss[var][verdict] += 1
            
    print("\n============================================================")
    print("ABLATION STUDY SUMMARY")
    print("============================================================")
    print("Friedman Average Ranks:")
    for name, rank in friedman_res["ranking_order"]:
        print(f"  {name:22s}: {rank:.3f}")
        
    print("\nWilcoxon Signed-Rank Test (RMA-EA (Full) vs Ablated Variants):")
    for var, counts in win_tie_loss.items():
        print(f"  Full vs {var:22s}: [+] {counts['+']} wins, [~] {counts['~']} ties, [-] {counts['-']} losses")
        
    output_data = {
        "dim": dim,
        "n_runs": n_runs,
        "max_fes": max_fes,
        "variants": variants,
        "problems": problem_names,
        "summary_table": summary_table,
        "all_errors": all_errors,
        "friedman_results": {
            "average_ranks": friedman_res["average_ranks"],
            "ranking_order": friedman_res["ranking_order"],
            "chi2_stat": friedman_res["chi2_stat"],
            "p_value": friedman_res["p_value"]
        },
        "wilcoxon_comp": wilcoxon_comp,
        "win_tie_loss": win_tie_loss
    }
    
    output_path = os.path.join(output_dir, f"ablation_results_D{dim}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAblation results successfully saved to: {output_path}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RMA-EA Ablation Experiments")
    parser.add_argument("--dim", type=int, default=10, help="Problem dimension")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per function")
    parser.add_argument("--max_fes", type=int, default=20000, help="Max evaluations per run")
    args = parser.parse_args()
    
    run_ablation_study(dim=args.dim, n_runs=args.runs, max_fes=args.max_fes)
