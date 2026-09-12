# RMA-EA: Riemannian Manifold Adaptive Evolutionary Algorithm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Target: IEEE TEVC](https://img.shields.io/badge/Target-IEEE%20TEVC%20%2F%20SWEVO-orange.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4235)

A mathematically grounded continuous evolutionary optimizer that endows an L-SHADE skeleton with an **online Riemannian metric on the Symmetric Positive Definite (SPD) manifold** $\mathcal{S}_{++}^D$.

The elite population's second-order structure is carried as a point on $\mathcal{S}_{++}^D$ under the **Log-Euclidean metric**, and its inverse (a *cometric*) is streamed into the variation operators as a time-varying, rotation-equivariant step-size field. The design targets three classical failure modes of differential evolution — coordinate rotation sensitivity, ill-conditioned narrow ravines, and deceptive multi-funnel multimodality — **without invoking biological metaphors and without adding a single tuned hyperparameter to the L-SHADE baseline**.

---

## Key Algorithmic Innovations

Let $C_t = G_t^{-1}$ denote the online cometric estimated from the elite sample at generation $t$.

1. **Riemannian Metric Flow on $\mathcal{S}_{++}^D$ (Log-Euclidean).** Elite second moments are embedded on the SPD manifold and updated by a Robbins-Monro recursion in the matrix-log domain, with **Marchenko-Pastur Bayesian shrinkage**
   $$\rho = \frac{D}{\mu + D}, \qquad C_t \leftarrow (1-\rho)\,\widehat{C}_t + \rho\,I,$$
   toward the isotropic metric. This guarantees strict rotation equivariance, removes the Euclidean "swelling effect", and keeps the diffusion operator rank-complete in all $D$ directions even after population collapse.

2. **Dual-Chart Atlas with Operator-Decoupled Bayesian Arbitration.** Two charts are maintained over the same search space: the *canonical* (Euclidean) chart and the *Riemannian principal-tangent* chart, in which crossover acts as a tangent projection $v = x + U\,[\,m \circ (U^\top d)\,]$. Chart selection and diffusion selection are governed by **two independent posteriors** updated from different credit signals:
   - $p_{\text{rot}}$ — crossover chart, credited by **survival counts** (mode-selection quality);
   - $p_{\text{diff}}$ — diffusion chart, credited by **improvement mass** $r = d_{\text{diff}}/(n_{\text{eval}}+1)$ (exploration productivity, scale-invariant).

   Decoupling the two is essential: a single shared posterior couples crossover and diffusion and collapses narrow-valley convergence.

3. **Geodesic Diffusion Mutation (GDM).** In the tangent chart the donor becomes
   $$v = x + F\,(x_{\text{pbest}} - x) + F\,g, \qquad g \sim \mathcal{N}\!\left(0,\ \sigma_{\text{pop}}^{2}\,C_t\right),\quad \sigma_{\text{pop}} = \sqrt{\tfrac{1}{D}\operatorname{Tr} C_{\text{emp}}},$$
   which injects a metric-shaped, zero-hyperparameter diffusion channel precisely where the difference vector degenerates after population contraction.

4. **Atlas Renewal under Chart Exhaustion.** When $H = $ `memory_size` consecutive generations yield no improvement, the atlas is declared exhausted: the sampler **re-initializes on the initial measure of the manifold (uniform over the domain)**, the incumbent leaves the population and is retained aside as the monotone record, the exhausted archive is released, and the cometric resets to the isotropic metric. This is deliberately *not* a local restart: in $D$ dimensions a Gaussian restart cloud centred on the exhausted attractor carries **exponentially vanishing** mass toward any other basin ($\sim e^{-10.8}$ at $D=30$), so renewal must return to the unbiased initial measure, and no greedy anchor may veto it.

5. **Low-Rank $O(k^2 D)$ Scalability.** Truncated spectral projection avoids the $O(D^3)$ eigendecomposition bottleneck, scaling to $D \ge 30, 50, 100$.

---

## Project Structure

```
.
├── src/
│   ├── rma_ea/
│   │   ├── manifold.py       # SPD manifold, Log-Euclidean metric, Riemannian metric flow
│   │   ├── operators.py      # Dual-chart mutation/crossover, decoupled Bayesian parameter memory
│   │   ├── algorithm.py      # RMA-EA core engine: L-SHADE skeleton + atlas renewal
│   │   └── landscape.py      # [Legacy] passive ruggedness sensor -- NOT used in the v3.x loop
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
├── tests/                    # 27 Pytest unit tests (100% passing)
├── experiments/
│   ├── run_cec_experiments.py   # Multi-run CEC benchmark driver
│   ├── run_ablation.py          # Ablation study driver
│   ├── validate_round{1,2,3,5}.py  # Same-seed A/B harnesses for each design iteration
│   ├── diagnose_f10.py, repro_f10_seed.py,
│   ├── escape_geometry_f10.py, attraction_f10.py   # Failure-mode diagnostics (D30 F10)
│   └── results/                 # JSON logs: current benchmark + per-round validations
└── docs/                        # Design specs and implementation plans
```

> **Implementation note.** `landscape.py` (the passive ruggedness sensor of earlier versions) is retained for ablation archaeology but is **not** invoked by the v3.x search loop; the `landscape_active` constructor flag is vestigial. All reported results come from the metric-flow / dual-chart / GDM / atlas-renewal machinery described above.

---

## Quickstart

### Development setup (recommended)

The repository ships with **no global installs**. Everything lives in a project-local virtual environment named `rma_ea/`, so your base Python is never touched.

```bash
# Create the venv from any Python 3.10+ (git-ignored, see .gitignore).
python -m venv rma_ea

# Activate it.  Windows PowerShell:
rma_ea\Scripts\Activate.ps1
# Windows cmd / Git Bash:
source rma_ea/Scripts/activate
# macOS / Linux:
source rma_ea/bin/activate

# Install the project in editable mode together with the runtime + dev deps.
pip install -e ".[dev]"
```

The `-e` flag means editing any file under `src/` is picked up immediately; no rebuild step is required.

#### Using the walk-through notebook

`pip install -e ".[dev]"` pulls in `ipykernel`, which is what VS Code and Jupyter need to start a kernel. To make the environment show up in VS Code's kernel picker, register it once:

```bash
python -m ipykernel install --user --name rma_ea --display-name "RMA-EA (rma_ea venv, Python 3.13)"
```

Then open `docs/rma_ea_walkthrough.ipynb` and pick **"RMA-EA (rma_ea venv, Python 3.13)"** from the kernel selector (top-right). If it does not appear, run *Python: Clear Cache and Reload Window* from the command palette.

> **Why this step exists.** A bare `python -m venv` contains no `ipykernel`, so the environment exists on disk but cannot serve a notebook kernel — VS Code will not list it. Installing `ipykernel` and registering a kernelspec is what makes it discoverable.

> **Naming note.** The virtual environment is called `rma_ea`, the same name as the source package `src/rma_ea/`. This is safe because `.gitignore` anchors the ignore rule as `/rma_ea/` (with a leading slash) — the bare pattern `rma_ea/` would also match `src/rma_ea/` and silently drop the source package from version control. At import time the regular package `src/rma_ea/` (which has `__init__.py`) always takes precedence over the venv directory, which is only a namespace package.

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
python experiments/run_cec_experiments.py --dim 10 --runs 30 --workers 8
python experiments/run_cec_experiments.py --dim 30 --runs 30 --workers 8
```
Iteration budget is standardized at 1000 (D10) / 1500 (D30); the seed schedule
`seed = 10000 + problem_index * 500 + run_index` is identical across all compared algorithms.

### 2. Ablation Study
```bash
python experiments/run_ablation.py --dim 10 --runs 20 --max_fes 20000
```

### 3. Design-Iteration Validation (same-seed A/B)
```bash
python experiments/validate_round5.py   # atlas renewal vs. prior version, 20 problems x 30 runs
```

### 4. Generate Publication-Quality Figures
```bash
python -c "import sys; sys.path.insert(0, 'src'); from visualization.plot_concept import generate_figure_1; from visualization.plots import plot_convergence_panels, plot_radar_performance, plot_ablation_bar_chart, plot_search_trajectories_2d; generate_figure_1(); plot_convergence_panels('experiments/results/cec_results_D10.json'); plot_radar_performance('experiments/results/cec_results_D10.json'); plot_ablation_bar_chart('experiments/results/ablation_results_D10.json'); plot_search_trajectories_2d()"
```

---

## Statistical Performance Highlights (Version 3.4)

Protocol: 10 functions x 30 runs, iteration-standardized (D10: 1000 iters, D30: 1500 iters), identical seed schedule across all algorithms.

- **D10 Friedman Ranking** ($\chi_F^2 = 22.41$, $p = 5.4 \times 10^{-5}$):
  1. **RMA-EA (Proposed)**: **1.350**
  2. L-SHADE (CEC Competition Winner): 2.100
  3. Standard DE: 2.550
  4. CMA-ES: 4.000
- **D30 Friedman Ranking** ($\chi_F^2 = 22.68$, $p = 4.7 \times 10^{-5}$):
  1. **RMA-EA (Proposed)**: **1.500**
  2. L-SHADE: 1.800
  3. Standard DE: 2.700
  4. CMA-ES: 4.000
- **Wilcoxon Signed-Rank vs. L-SHADE** ($\alpha = 0.05$): **zero losses in all 20 problem-dimension pairs** (D10: 3/7/0, D30: 3/7/0). Significant wins: D10 F4 ($p = 0.0001$), F5 ($p = 0.0009$), F8 ($p < 10^{-4}$); D30 F4 ($p < 10^{-4}$), F5 ($p = 0.0004$), F9 ($p = 0.0006$). All remaining ties are exact zeros for both algorithms.
- **vs. CMA-ES**: 10-0 clean sweep at both dimensions (Holm-corrected $p < 10^{-4}$).
- **Multimodal highlights (mean error vs. L-SHADE)**:
  - D10 F4 (Rastrigin): $1.92$ vs $3.08$; D10 F5 (Schaffer F6): $1.51$ vs $2.05$; D10 F9 (Hybrid): $0$ vs $0.37$; D10 F7 (Schwefel): $0$ vs $7.24$.
  - D30 F4: $9.86$ vs $15.54$; D30 F5: $10.15$ vs $10.71$; D30 F9: $15.12$ vs $24.17$.
  - D30 F10 (Composition): $0$ vs $0$ -- all 30 runs reach the global basin; the chart-exhaustion atlas renewal recovers the two previously trapped runs (Schwefel-dominated local basin, 132.6 error) via an unbiased uniform re-initialization of the sampler with the incumbent retained aside.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{wang2026rmaea,
  author    = {Jinpeng Wang},
  title     = {RMA-EA: Riemannian Manifold Adaptive Evolutionary Algorithm},
  version   = {3.4},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/JinPengWang/RMA-EA}
}
```

---

## License

Released under the [MIT License](LICENSE).
