# RMA-EA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, test, experimentally evaluate, and document RMA-EA (Riemannian Manifold & Landscape-Ruggedness Adaptive Evolutionary Algorithm) with CEC benchmarks and IEEE TEVC paper draft.

**Architecture:** Python 3.12 implementation with modular packages: `src/rma_ea/` (manifold, landscape, operators, algorithm), `src/benchmarks/` (CEC2017/2022, classic), `src/baselines/` (DE, L-SHADE, CMA-ES), `src/analysis/` (Wilcoxon, Friedman, logging), `src/visualization/` (Nature/IEEE plots), and `paper/` (manuscript and figures).

**Tech Stack:** Python 3.12, NumPy, SciPy, Matplotlib, Pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-rma-ea-design.md`

## Global Constraints
- Target platform: Windows, Python 3.12+
- All matrix operations on SPD manifold must enforce numerical symmetry and positive-definiteness ($\epsilon = 10^{-12}$).
- Pure NumPy + SciPy for core algorithms to ensure seamless cross-platform execution without requiring C compilers.
- Rigorous TDD: tests written and passing for each core module before proceeding.

---

### Task 1: Project Scaffolding & Pytest Setup
**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create project configuration `pyproject.toml`**
- [ ] **Step 2: Verify pytest runs on empty suite**
- [ ] **Step 3: Commit scaffolding**

---

### Task 2: SPD Riemannian Manifold Engine (`src/rma_ea/manifold.py`)
**Files:**
- Create: `src/rma_ea/manifold.py`
- Test: `tests/test_manifold.py`

**Interfaces:**
- `matrix_log(C: np.ndarray, eps: float = 1e-12) -> np.ndarray`
- `matrix_exp(M: np.ndarray) -> np.ndarray`
- `log_euclidean_distance(C1: np.ndarray, C2: np.ndarray) -> float`
- `geodesic(C1: np.ndarray, C2: np.ndarray, tau: float) -> np.ndarray`
- `update_riemannian_barycenter(C_prev: np.ndarray, C_new: np.ndarray, learning_rate: float) -> np.ndarray`
- `low_rank_covariance_decompose(samples: np.ndarray, weights: np.ndarray, rank_k: int) -> Tuple[np.ndarray, np.ndarray, float]`

- [ ] **Step 1: Write unit tests for manifold operations**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Implement manifold operations**
- [ ] **Step 4: Verify all manifold tests pass**
- [ ] **Step 5: Commit**

---

### Task 3: Online Landscape Ruggedness Sensor (`src/rma_ea/landscape.py`)
**Files:**
- Create: `src/rma_ea/landscape.py`
- Test: `tests/test_landscape.py`

**Interfaces:**
- `class LandscapeRuggednessSensor`:
  - `__init__(self, dim: int, history_len: int = 20, kappa: float = 2.0)`
  - `sense_directional_ruggedness(self, pop: np.ndarray, fitness: np.ndarray, eig_vecs: np.ndarray, eig_vals: np.ndarray, eval_func: Callable) -> float`
  - `update_index(self, ruggedness_val: float) -> float`
  - `get_exploration_probability(self) -> float`

- [ ] **Step 1: Write unit tests for landscape sensor**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Implement LandscapeRuggednessSensor**
- [ ] **Step 4: Verify tests pass**
- [ ] **Step 5: Commit**

---

### Task 4: Dual-Channel Mutation, Crossover & Parameter Memory (`src/rma_ea/operators.py`)
**Files:**
- Create: `src/rma_ea/operators.py`
- Test: `tests/test_operators.py`

**Interfaces:**
- `class ParameterMemory`:
  - `sample_parameters(self, size: int) -> Tuple[np.ndarray, np.ndarray]`
  - `update_memory(self, successful_F: np.ndarray, successful_Cr: np.ndarray, fitness_improvements: np.ndarray)`
- `class DualChannelMutation`:
  - `mutate_geodesic_drift(...) -> np.ndarray`
  - `mutate_anisotropic_contraction(...) -> np.ndarray`
- `binomial_crossover(target: np.ndarray, donor: np.ndarray, Cr: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write unit tests for operators**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Implement operators and parameter memory**
- [ ] **Step 4: Verify tests pass**
- [ ] **Step 5: Commit**

---

### Task 5: RMA-EA Core Optimization Algorithm (`src/rma_ea/algorithm.py`)
**Files:**
- Create: `src/rma_ea/algorithm.py`
- Test: `tests/test_algorithm.py`

**Interfaces:**
- `class RMA_EA`:
  - `__init__(self, problem, max_fes: int, pop_init: int = 100, pop_min: int = 4, p_best_rate: float = 0.11, ...)`
  - `optimize() -> OptimizationResult`

- [ ] **Step 1: Write unit tests for algorithm optimization on Sphere, Rosenbrock, Rastrigin**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Implement RMA-EA with LPSR and archive**
- [ ] **Step 4: Verify tests pass and algorithm reaches global optima**
- [ ] **Step 5: Commit**

---

### Task 6: Benchmark Evaluation Suite (`src/benchmarks/`)
**Files:**
- Create: `src/benchmarks/base.py`
- Create: `src/benchmarks/classic.py`
- Create: `src/benchmarks/cec2017.py`
- Create: `src/benchmarks/cec2022.py`
- Test: `tests/test_benchmarks.py`

- [ ] **Step 1: Write unit tests for benchmark suites**
- [ ] **Step 2: Implement benchmarks with shift & rotation matrices**
- [ ] **Step 3: Verify all test benchmarks pass**
- [ ] **Step 4: Commit**

---

### Task 7: Baseline Algorithms (`src/baselines/`)
**Files:**
- Create: `src/baselines/de.py` (Standard DE)
- Create: `src/baselines/lshade.py` (L-SHADE state-of-the-art CEC baseline)
- Create: `src/baselines/cmaes.py` (CMA-ES baseline)
- Test: `tests/test_baselines.py`

- [ ] **Step 1: Implement DE, L-SHADE, and CMA-ES baselines**
- [ ] **Step 2: Verify baselines pass unit tests**
- [ ] **Step 3: Commit**

---

### Task 8: Statistical Analysis & Experiment Framework (`src/analysis/`, `experiments/`)
**Files:**
- Create: `src/analysis/statistics.py` (Wilcoxon signed-rank test, Friedman test)
- Create: `experiments/run_cec_experiments.py`
- Create: `experiments/run_ablation.py`

- [ ] **Step 1: Implement statistical testing modules**
- [ ] **Step 2: Run multi-run comparative experiments on benchmarks**
- [ ] **Step 3: Run ablation study (RMA-EA vs w/o Manifold vs w/o LRS)**
- [ ] **Step 4: Output statistical significance tables ($p$-values, Wilcoxon +/-/=, Friedman ranks)**

---

### Task 9: Publication-Quality Scientific Visualizations (`src/visualization/plots.py`)
**Files:**
- Create: `src/visualization/plots.py`
- Output: `paper/figures/fig1_concept_manifold.pdf/png`
- Output: `paper/figures/fig2_convergence_curves.pdf/png`
- Output: `paper/figures/fig3_radar_comparison.pdf/png`
- Output: `paper/figures/fig4_ablation_study.pdf/png`
- Output: `paper/figures/fig5_search_trajectories_2d.pdf/png`

- [ ] **Step 1: Generate high-resolution convergence curves with shaded IQR**
- [ ] **Step 2: Generate 2D landscape trajectory comparisons**
- [ ] **Step 3: Generate radar charts and ablation figures**

---

### Task 10: Academic Paper Manuscript Draft (`paper/`)
**Files:**
- Create: `paper/manuscript.md`
- Create: `paper/main.tex` (IEEE Transactions on Evolutionary Computation style)
- Create: `paper/references.bib`

- [ ] **Step 1: Write complete Sections (Abstract, Introduction, Background, Proposed RMA-EA, Experiments & Statistical Analysis, Ablation Studies, Conclusion)**
- [ ] **Step 2: Verify citations and format compliance**
- [ ] **Step 3: Final project commit**
