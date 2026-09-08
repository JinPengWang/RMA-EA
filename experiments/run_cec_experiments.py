"""CEC Benchmark Multi-Run Comparative Experiment Suite.

Executes comparative evaluations across RMA-EA and baseline algorithms (L-SHADE, CMA-ES, StandardDE)
on the CEC benchmark suite, computing Wilcoxon signed-rank tests, Friedman average rankings,
and saving all convergence traces for plotting.
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
from baselines.lshade import LSHADE
from baselines.cmaes import CMAES
from benchmarks.cec_suite import get_benchmark_suite
from analysis.statistics import (
    summarize_run_statistics,
    wilcoxon_test,
    friedman_test,
    holm_posthoc
)


def run_benchmark_experiments(
    dim: int = 10,
    n_runs: int = 20,
    max_fes: int = 20000,
    output_dir: str = "experiments/results"
) -> Dict[str, Any]:
    """Execute full benchmark suite comparison."""
    os.makedirs(output_dir, exist_ok=True)
    suite = get_benchmark_suite(dim=dim)
    
    algorithms = ["RMA-EA", "L-SHADE", "CMA-ES", "StandardDE"]
    problem_names = [f.name for f in suite]
    
    # Store error outcomes: algo -> problem -> list of final errors
    all_errors: Dict[str, Dict[str, List[float]]] = {
        algo: {prob: [] for prob in problem_names} for algo in algorithms
    }
    
    # Store convergence traces (first run of each algorithm for plotting)
    convergence_traces: Dict[str, Dict[str, Dict[str, List]]] = {
        prob: {} for prob in problem_names
    }
    
    # Store landscape sensor traces for RMA-EA
    ruggedness_traces: Dict[str, List[float]] = {}
    
    print(f"============================================================")
    print(f"Starting CEC Benchmark Experiment Suite: Dim={dim}, Runs={n_runs}, MaxFES={max_fes}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"============================================================")
    
    start_time = time.time()
    
    for prob_idx, func in enumerate(suite):
        p_name = func.name
        print(f"\nEvaluating Problem [{prob_idx + 1}/{len(suite)}]: {p_name} ({func.category})")
        
        for run_idx in range(n_runs):
            seed = 10000 + prob_idx * 500 + run_idx
            
            # 1. RMA-EA
            opt_rma = RMA_EA(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                seed=seed
            )
            res_rma = opt_rma.optimize()
            err_rma = max(0.0, res_rma.best_f - func.bias)
            all_errors["RMA-EA"][p_name].append(float(err_rma))
            
            # 2. L-SHADE
            opt_lshade = LSHADE(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                seed=seed
            )
            res_lshade = opt_lshade.optimize()
            err_lshade = max(0.0, res_lshade.best_f - func.bias)
            all_errors["L-SHADE"][p_name].append(float(err_lshade))
            
            # 3. CMA-ES
            opt_cma = CMAES(
                objective_func=func,
                dim=dim,
                lower_bound=func.bounds[0],
                upper_bound=func.bounds[1],
                max_fes=max_fes,
                seed=seed
            )
            res_cma = opt_cma.optimize()
            err_cma = max(0.0, res_cma.best_f - func.bias)
            all_errors["CMA-ES"][p_name].append(float(err_cma))
            
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
            err_de = max(0.0, res_de.best_f - func.bias)
            all_errors["StandardDE"][p_name].append(float(err_de))
            
            # Save convergence trajectory from run 0
            if run_idx == 0:
                convergence_traces[p_name]["RMA-EA"] = {
                    "fes": res_rma.history_fes,
                    "errors": [max(0.0, f - func.bias) for f in res_rma.history_fitness]
                }
                convergence_traces[p_name]["L-SHADE"] = {
                    "fes": res_lshade.history_fes,
                    "errors": [max(0.0, f - func.bias) for f in res_lshade.history_fitness]
                }
                convergence_traces[p_name]["CMA-ES"] = {
                    "fes": res_cma.history_fes,
                    "errors": [max(0.0, f - func.bias) for f in res_cma.history_fitness]
                }
                convergence_traces[p_name]["StandardDE"] = {
                    "fes": res_de.history_fes,
                    "errors": [max(0.0, f - func.bias) for f in res_de.history_fitness]
                }
                ruggedness_traces[p_name] = res_rma.history_ruggedness
                
        # Print summary for this problem
        m_rma = np.mean(all_errors["RMA-EA"][p_name])
        m_lshade = np.mean(all_errors["L-SHADE"][p_name])
        m_cma = np.mean(all_errors["CMA-ES"][p_name])
        m_de = np.mean(all_errors["StandardDE"][p_name])
        print(f"  Mean Errors -> RMA-EA: {m_rma:.2e} | L-SHADE: {m_lshade:.2e} | CMA-ES: {m_cma:.2e} | DE: {m_de:.2e}")
        
    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed:.2f} seconds.")
    
    # Statistical analysis
    summary_table: Dict[str, Dict[str, Dict[str, float]]] = {
        prob: {} for prob in problem_names
    }
    
    # Wilcoxon comparisons of RMA-EA vs competitors
    wilcoxon_results: Dict[str, Dict[str, Any]] = {
        comp: {} for comp in ["L-SHADE", "CMA-ES", "StandardDE"]
    }
    win_tie_loss: Dict[str, Dict[str, int]] = {
        comp: {"+": 0, "~": 0, "-": 0} for comp in ["L-SHADE", "CMA-ES", "StandardDE"]
    }
    
    # Mean error matrix for Friedman test (problems x algorithms)
    mean_error_matrix = np.zeros((len(problem_names), len(algorithms)))
    
    for p_idx, prob in enumerate(problem_names):
        rma_runs = np.array(all_errors["RMA-EA"][prob])
        for a_idx, algo in enumerate(algorithms):
            runs = np.array(all_errors[algo][prob])
            stats_dict = summarize_run_statistics(runs)
            summary_table[prob][algo] = stats_dict
            mean_error_matrix[p_idx, a_idx] = stats_dict["mean"]
            
        for comp in ["L-SHADE", "CMA-ES", "StandardDE"]:
            comp_runs = np.array(all_errors[comp][prob])
            verdict, stat, p_val = wilcoxon_test(rma_runs, comp_runs)
            wilcoxon_results[comp][prob] = {
                "verdict": verdict,
                "stat": stat,
                "p_val": p_val
            }
            win_tie_loss[comp][verdict] += 1
            
    # Friedman test
    friedman_res = friedman_test(mean_error_matrix, algorithms)
    holm_res = holm_posthoc(friedman_res, control_name="RMA-EA", n_problems=len(problem_names))
    
    print("\n============================================================")
    print("STATISTICAL TESTING SUMMARY (Control: RMA-EA)")
    print("============================================================")
    print("Friedman Average Ranks (1.0 = Best):")
    for name, rank in friedman_res["ranking_order"]:
        print(f"  {name:12s}: {rank:.3f}")
    print(f"Friedman Chi^2: {friedman_res['chi2_stat']:.3f}, p-value: {friedman_res['p_value']:.4e}")
    
    print("\nHolm Post-Hoc Test Comparisons:")
    for comp in holm_res:
        sig_str = "SIGNIFICANT" if comp["significant"] else "NOT SIGNIFICANT"
        print(f"  {comp['comparison']:25s}: z={comp['z_value']:.3f}, p={comp['p_value']:.4e} ({sig_str})")
        
    print("\nWilcoxon Signed-Rank Test Win / Tie / Loss (RMA-EA vs Competitors):")
    for comp, counts in win_tie_loss.items():
        print(f"  RMA-EA vs {comp:12s}: [+] {counts['+']} wins, [~] {counts['~']} ties, [-] {counts['-']} losses")
        
    output_data = {
        "dim": dim,
        "n_runs": n_runs,
        "max_fes": max_fes,
        "algorithms": algorithms,
        "problems": problem_names,
        "summary_table": summary_table,
        "all_errors": all_errors,
        "wilcoxon_results": wilcoxon_results,
        "win_tie_loss": win_tie_loss,
        "friedman_results": {
            "average_ranks": friedman_res["average_ranks"],
            "ranking_order": friedman_res["ranking_order"],
            "chi2_stat": friedman_res["chi2_stat"],
            "p_value": friedman_res["p_value"]
        },
        "holm_results": holm_res,
        "convergence_traces": convergence_traces,
        "ruggedness_traces": ruggedness_traces
    }
    
    output_path = os.path.join(output_dir, f"cec_results_D{dim}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nExperiment results successfully saved to: {output_path}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CEC Benchmark Experiments")
    parser.add_argument("--dim", type=int, default=10, help="Problem dimension")
    parser.add_argument("--runs", type=int, default=20, help="Number of runs per function")
    parser.add_argument("--max_fes", type=int, default=20000, help="Max evaluations per run")
    args = parser.parse_args()
    
    run_benchmark_experiments(dim=args.dim, n_runs=args.runs, max_fes=args.max_fes)
