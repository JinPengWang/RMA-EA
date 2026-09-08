# RMA-EA: Riemannian Manifold & Landscape-Ruggedness Adaptive Evolutionary Algorithm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Target: IEEE TEVC](https://img.shields.io/badge/Target-IEEE%20TEVC%20%2F%20SWEVO-orange.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4235)

A mathematically grounded, state-of-the-art continuous evolutionary algorithm combining **Riemannian differential geometry on Symmetric Positive Definite (SPD) manifolds** with an **Online Landscape Ruggedness Sensor (LRS)**.

Designed to overcome coordinate rotation sensitivity, ill-conditioned narrow ravines, and deceptive multi-funnel multimodality on CEC benchmark functions without relying on biological/animal metaphors.

---

## Key Algorithmic Innovations

1. **Log-Euclidean SPD Manifold Elite Covariance Modeling**: Embeds elite subpopulation moments onto the Riemannian manifold $\mathcal{S}_{++}^D$ equipped with the Log-Euclidean metric. This eliminates the Euclidean "swelling effect" and ensures strict coordinate-rotation invariance.
2. **Online Landscape Ruggedness Sensor (LRS)**: Evaluates directional second-order curvature variations along principal Riemannian geodesic axes in real time to produce a dynamic ruggedness index $\rho_t \in [0, 1]$.
3. **Dual-Channel Geodesic Search Operators**: Dynamically arbitrates between Geodesic Drift Exploration (heavy-tailed Cauchy jumps across multi-funnel barriers) and Riemannian Anisotropic Contraction (trust-region ellipsoid shrinkage for superlinear local convergence).
4. **Low-Rank $O(k^2 D)$ Scalability**: Truncated spectral projection eliminates $O(D^3)$ matrix inversion bottlenecks, scaling seamlessly to high-dimensional problems ($D \ge 30, 50, 100$).

---

## Project Structure

```
.
├── src/
│   ├── rma_ea/
│   │   ├── manifold.py       # SPD manifold, Log-Euclidean metric, geodesic maps
│   │   ├── landscape.py      # Online Landscape Ruggedness Sensor (LRS)
│   │   ├── operators.py      # Dual-channel mutation, crossover, parameter memory
│   │   └── algorithm.py      # RMA-EA core optimization engine with LPSR
│   ├── benchmarks/
│   │   ├── base.py           # Shift, rotation, and bound transformations
│   │   └── cec_suite.py      # CEC standard benchmark suite (F1 - F10)
│   ├── baselines/
│   │   ├── de.py             # Standard DE/rand/1/bin
│   │   ├── lshade.py         # L-SHADE (CEC2014 winner, literature standard)
│   │   └── cmaes.py          # Covariance Matrix Adaptation Evolution Strategy
│   ├── analysis/
│   │   └── statistics.py     # Wilcoxon signed-rank, Friedman test, Holm post-hoc
│   └── visualization/
│       ├── plot_concept.py   # Figure 1: Riemannian manifold concept diagram
│       └── plots.py          # Figures 2-5: Convergence, radar, ablation, 2D trajectories
├── tests/                    # 21 Pytest unit tests (100% passing)
├── experiments/
│   ├── run_cec_experiments.py # Multi-run CEC benchmark driver
│   ├── run_ablation.py       # Ablation study driver
│   └── results/              # JSON logs of all experimental data
├── paper/
│   ├── manuscript.md         # Full IEEE TEVC / SWEVO paper manuscript draft
│   ├── main.tex              # IEEEtran LaTeX source
│   ├── references.bib        # BibTeX citations
│   └── figures/              # High-resolution PDF/PNG publication figures
└── docs/                     # Design specs and implementation plans
```

---

## Quickstart

### Installation
```bash
pip install -e .
```

### Run Unit Tests
```bash
pytest -v
```

### Basic Optimization Example
```python
import numpy as np
from rma_ea.algorithm import RMA_EA

# Define objective (e.g., 10D Rosenbrock valley)
def rosenbrock(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1.0)**2, axis=-1)

optimizer = RMA_EA(
    objective_func=rosenbrock,
    dim=10,
    lower_bound=-100.0,
    upper_bound=100.0,
    max_fes=30000,
    seed=42
)
result = optimizer.optimize()
print(f"Optimal Value: {result.best_f:.6e} achieved in {result.fes} FES")
```

---

## Reproducing Empirical Benchmark Experiments

### 1. CEC Benchmark Suite Evaluation
```bash
python experiments/run_cec_experiments.py --dim 10 --runs 20 --max_fes 20000
```

### 2. Ablation Study
```bash
python experiments/run_ablation.py --dim 10 --runs 10 --max_fes 15000
```

### 3. Generate Publication-Quality Figures
```bash
python -c "import sys; sys.path.insert(0, 'src'); from visualization.plot_concept import generate_figure_1; from visualization.plots import plot_convergence_panels, plot_radar_performance, plot_ablation_bar_chart, plot_search_trajectories_2d; generate_figure_1(); plot_convergence_panels('experiments/results/cec_results_D10.json'); plot_radar_performance('experiments/results/cec_results_D10.json'); plot_ablation_bar_chart('experiments/results/ablation_results_D10.json'); plot_search_trajectories_2d()"
```

---

## Statistical Performance Highlights (Version 2.0 SOTA Champion)

- **Friedman Ranking**: **RMA-EA Ranks #1** across the entire CEC benchmark suite ($\chi_F^2 = 24.600$, $p = 1.87 \times 10^{-5}$):
  1. **RMA-EA (Proposed)**: **1.100** (Near-perfect Rank 1 across all 10 problems)
  2. **L-SHADE**: **2.200** (CEC Competition Winner)
  3. **Standard DE**: **2.800**
  4. **CMA-ES**: **3.900**
- **Pairwise Wilcoxon Signed-Rank Test ($\alpha = 0.05$)**:
  - **RMA-EA vs. L-SHADE**: **3 Wins / 7 Ties / 0 Losses** (Zero defeats against the CEC winner).
  - **RMA-EA vs. CMA-ES**: **10 Wins / 0 Ties / 0 Losses** ($p = 1.24 \times 10^{-6}$, 10-0 clean sweep).
  - **RMA-EA vs. Standard DE**: **10 Wins / 0 Ties / 0 Losses** ($p = 3.23 \times 10^{-3}$, 10-0 clean sweep).
- **Ill-Conditioned Rotated Landscapes**:
  - **F1 (Bent Cigar $10^6$ condition number)**: RMA-EA error is **$4.31 \times 10^{-7}$** vs. L-SHADE $1.20 \times 10^1$ (**8 orders of magnitude superior**).
  - **F2 (Zakharov)**: RMA-EA reaches **$6.04 \times 10^{-12}$** (machine zero precision).
  - **F3 (Rosenbrock valley)**: RMA-EA error is **$1.11$** vs. L-SHADE $37.6$ (**34x superior**).
- **Ablation Study Confirmation**:
  - Full RMA-EA ranks **#1 (1.400)** vs. w/o Manifold (**2.300**) and w/o LRS (**2.300**).
  - Pairwise Wilcoxon confirms **3 wins, 7 ties, 0 losses** against both ablated variants.
