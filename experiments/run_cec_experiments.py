"""CEC Benchmark Multi-Run Comparative Experiment Suite.

Executes comparative evaluations across RMA-EA and baseline algorithms (L-SHADE, CMA-ES, StandardDE)
on the CEC benchmark suite, computing Wilcoxon signed-rank tests, Friedman average rankings,
and saving all convergence traces for plotting.

Strictly adheres to IEEE CEC benchmark competition protocol:
- Maximum function evaluations: MaxFES = 10,000 * D.
- Statistical significance testing via Wilcoxon signed-rank test with Holm post-hoc correction.
"""

import sys
import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
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


def _worker_single_run(args: Tuple[int, int, int, int]) -> Tuple[int, int, Dict[str, float], Any]:
    """Worker process evaluating 4 algorithms on one problem and run seed."""
    prob_idx, run_idx, dim, max_fes = args
    suite = get_benchmark_suite(dim=dim)
    func = suite[prob_idx]
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
    err_rma = max(0.0, float(res_rma.best_f - func.bias))
    if err_rma < 1e-8:
        err_rma = 0.0
    
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
    err_lshade = max(0.0, float(res_lshade.best_f - func.bias))
    if err_lshade < 1e-8:
        err_lshade = 0.0
    
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
    err_cma = max(0.0, float(res_cma.best_f - func.bias))
    if err_cma < 1e-8:
        err_cma = 0.0
    
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
    err_de = max(0.0, float(res_de.best_f - func.bias))
    if err_de < 1e-8:
        err_de = 0.0
    
    trace_data = None
    if run_idx == 0:
        trace_data = {
            "RMA-EA": {
                "fes": res_rma.history_fes,
                "errors": [max(0.0, float(f - func.bias)) for f in res_rma.history_fitness]
            },
            "L-SHADE": {
                "fes": res_lshade.history_fes,
                "errors": [max(0.0, float(f - func.bias)) for f in res_lshade.history_fitness]
            },
            "CMA-ES": {
                "fes": res_cma.history_fes,
                "errors": [max(0.0, float(f - func.bias)) for f in res_cma.history_fitness]
            },
            "StandardDE": {
                "fes": res_de.history_fes,
                "errors": [max(0.0, float(f - func.bias)) for f in res_de.history_fitness]
            },
            "ruggedness": res_rma.history_ruggedness
        }
        
    errors = {
        "RMA-EA": err_rma,
        "L-SHADE": err_lshade,
        "CMA-ES": err_cma,
        "StandardDE": err_de
    }
    return prob_idx, run_idx, errors, trace_data


def run_benchmark_experiments(
    dim: int = 10,
    n_runs: int = 10,
    max_fes: int = None,
    n_workers: int = 8,
    output_dir: str = "experiments/results"
) -> Dict[str, Any]:
    """Execute full benchmark suite comparison with multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    suite = get_benchmark_suite(dim=dim)
    
    if max_fes is None:
        max_fes = 10000 * dim
        
    algorithms = ["RMA-EA", "L-SHADE", "CMA-ES", "StandardDE"]
    problem_names = [f.name for f in suite]
    
    all_errors: Dict[str, Dict[str, List[float]]] = {
        algo: {prob: [0.0] * n_runs for prob in problem_names} for algo in algorithms
    }
    convergence_traces: Dict[str, Dict[str, Dict[str, List]]] = {
        prob: {} for prob in problem_names
    }
    ruggedness_traces: Dict[str, List[float]] = {}
    
    print(f"============================================================")
    print(f"Starting CEC Benchmark: Dim={dim}, Runs={n_runs}, MaxFES={max_fes}, Workers={n_workers}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"============================================================")
    
    start_time = time.time()
    tasks = []
    for prob_idx in range(len(suite)):
        for run_idx in range(n_runs):
            tasks.append((prob_idx, run_idx, dim, max_fes))
            
    completed_count = 0
    total_tasks = len(tasks)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(_worker_single_run, t): t for t in tasks}
        for future in as_completed(future_map):
            prob_idx, run_idx, errors, trace_data = future.result()
            p_name = problem_names[prob_idx]
            for algo, err in errors.items():
                all_errors[algo][p_name][run_idx] = err
                
            if trace_data is not None:
                for algo in algorithms:
                    convergence_traces[p_name][algo] = trace_data[algo]
                ruggedness_traces[p_name] = trace_data["ruggedness"]
                
            completed_count += 1
            if completed_count % max(1, total_tasks // 10) == 0 or completed_count == total_tasks:
                elapsed = time.time() - start_time
                print(f"Progress: [{completed_count}/{total_tasks}] tasks completed ({elapsed:.1f}s)")
                
    elapsed_total = time.time() - start_time
    print(f"\nAll {total_tasks} runs completed in {elapsed_total:.2f} seconds.")
    
    # Statistical analysis
    summary_table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for p_name in problem_names:
        summary_table[p_name] = {}
        for algo in algorithms:
            summary_table[p_name][algo] = summarize_run_statistics(all_errors[algo][p_name])
            
    wilcoxon_results: Dict[str, Dict[str, Any]] = {}
    win_tie_loss: Dict[str, Dict[str, int]] = {}
    
    for comp_algo in ["L-SHADE", "CMA-ES", "StandardDE"]:
        wilcoxon_results[comp_algo] = {}
        win_tie_loss[comp_algo] = {"+": 0, "~": 0, "-": 0}
        for p_name in problem_names:
            rma_errs = all_errors["RMA-EA"][p_name]
            comp_errs = all_errors[comp_algo][p_name]
            verdict, stat, p_val = wilcoxon_test(rma_errs, comp_errs)
            wilcoxon_results[comp_algo][p_name] = {
                "stat": float(stat),
                "p_value": float(p_val),
                "significance": verdict
            }
            win_tie_loss[comp_algo][verdict] += 1
            
    # Friedman test
    score_matrix = np.array([
        [summary_table[p][algo]["mean"] for algo in algorithms]
        for p in problem_names
    ])
    friedman_res = friedman_test(score_matrix, algorithms)
    holm_res = holm_posthoc(friedman_res, control_name="RMA-EA", n_problems=len(problem_names))
    
    print("\n" + "=" * 60)
    print("EXPERIMENTAL SUMMARY & STATISTICAL ANALYSIS")
    print("=" * 60)
    print("\nFriedman Average Rankings (lower is better):")
    for algo, rank in friedman_res["average_ranks"].items():
        print(f"  {algo:15s}: {rank:.4f}")
    print(f"  Chi-Square Statistic: {friedman_res['chi2_stat']:.4f}, p-value: {friedman_res['p_value']:.4e}")
    
    print("\nHolm's Post-Hoc Test vs Control (RMA-EA):")
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
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per function")
    parser.add_argument("--max_fes", type=int, default=None, help="Max evaluations (defaults to 10000*dim)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    run_benchmark_experiments(
        dim=args.dim,
        n_runs=args.runs,
        max_fes=args.max_fes,
        n_workers=args.workers
    )
