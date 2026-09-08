# RMA-EA: Riemannian Manifold & Landscape-Ruggedness Adaptive Evolutionary Algorithm
## Technical Design Specification

**Target Venue**: IEEE Transactions on Evolutionary Computation (IEEE TEVC) / Swarm and Evolutionary Computation (SWEVO)
**Author System**: DeepMind Antigravity Advanced Agentic Research Team
**Date**: 2026-09-08

---

## 1. Executive Summary & Problem Formulation

In continuous numerical optimization, real-world objective landscapes $f: \mathbb{R}^D \to \mathbb{R}$ are frequently plagued by non-separability, severe coordinate rotation, high condition numbers (ill-conditioned valleys), and deceptive multi-funnel basins (as instantiated in standard CEC2017 and CEC2022 benchmarks).

Standard evolutionary algorithms (such as Differential Evolution variants and Particle Swarm Optimizers) operate within an Euclidean coordinate framework. When landscapes are rotated or ill-conditioned, Euclidean differential vectors $\mathbf{x}_{r_1} - \mathbf{x}_{r_2}$ fail to align with the underlying low-dimensional fitness manifolds, precipitating premature stagnation or exponential search deceleration.

**RMA-EA (Riemannian Manifold & Landscape-Ruggedness Adaptive Evolutionary Algorithm)** introduces a rigorous geometric framework:
1. **Symmetric Positive Definite (SPD) Manifold $\mathcal{S}_{++}^D$ Elite Covariance Modeling**: Embeds elite population second-order moments onto the Riemannian manifold $\mathcal{S}_{++}^D$ equipped with the Log-Euclidean Riemannian metric. It computes intrinsic natural search geodesics with coordinate-invariance.
2. **Online Landscape Ruggedness Sensor (LRS)**: Probes local directional fitness variation online, calculating a dynamic ruggedness index $\rho_t \in [0, 1]$ to detect transitions between smooth valleys, deceptive multi-funnels, and rugged plateaus.
3. **Dual-Channel Geodesic Search Operators**: Dynamically arbitrates between Geodesic Drift Exploration (traversing non-Euclidean geodesics across basins) and Riemannian Anisotropic Contraction (exploitative refinement within valleys).
4. **Low-Rank $O(k^2 D)$ Geometric Projection**: Provides computational scalability up to $D=100$ without $O(D^3)$ matrix inversion bottlenecks.

---

## 2. Mathematical Foundation & Manifold Geometry

### 2.1 The SPD Manifold and the Log-Euclidean Metric
Let $\mathcal{S}_{++}^D = \{\mathbf{M} \in \mathbb{R}^{D \times D} : \mathbf{M} = \mathbf{M}^\top, \, \mathbf{v}^\top \mathbf{M} \mathbf{v} > 0, \, \forall \mathbf{v} \neq \mathbf{0}\}$ denote the cone of symmetric positive definite matrices.

Under the standard Euclidean metric, straight lines between matrices do not preserve the SPD geometry (the swelling effect) and curvature is ignored. RMA-EA employs the **Log-Euclidean Metric**:
For $\mathbf{C}_1, \mathbf{C}_2 \in \mathcal{S}_{++}^D$:
$$d_{\text{LE}}(\mathbf{C}_1, \mathbf{C}_2) = \|\log(\mathbf{C}_1) - \log(\mathbf{C}_2)\|_F$$
where $\log(\mathbf{C}) = \mathbf{U} \operatorname{diag}(\ln \lambda_1, \dots, \ln \lambda_D) \mathbf{U}^\top$ for spectral decomposition $\mathbf{C} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$.

### 2.2 Geodesic Formulation & Riemannian Barycenter
The unique geodesic curve $\gamma: [0, 1] \to \mathcal{S}_{++}^D$ connecting $\mathbf{C}_1$ and $\mathbf{C}_2$ is:
$$\gamma(\tau) = \exp\left( (1-\tau)\log(\mathbf{C}_1) + \tau \log(\mathbf{C}_2) \right)$$

Given historical elite covariance matrices $\{\mathbf{C}^{(1)}, \dots, \mathbf{C}^{(t)}\}$ with temporal decay weights $\beta_k$, the recursive Riemannian barycenter $\bar{\mathbf{C}}_{t}$ satisfies:
$$\log(\bar{\mathbf{C}}_t) = (1 - c_m) \log(\bar{\mathbf{C}}_{t-1}) + c_m \log(\mathbf{C}_{\mathcal{E}}^{(t)})$$
where $c_m \in (0, 1]$ is the manifold learning rate, and $\mathbf{C}_{\mathcal{E}}^{(t)}$ is the weighted covariance of the top $\mu = \lfloor p \cdot N \rfloor$ individuals:
$$\mathbf{m}^{(t)} = \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\mu}^{(t)}, \quad \mathbf{C}_{\mathcal{E}}^{(t)} = \sum_{i=1}^{\mu} w_i (\mathbf{x}_{i:\mu}^{(t)} - \mathbf{m}^{(t)})(\mathbf{x}_{i:\mu}^{(t)} - \mathbf{m}^{(t)})^\top + \epsilon \mathbf{I}_D$$

### 2.3 Low-Rank Subspace Projection ($O(k^2 D)$ Scalability)
For high-dimensional spaces ($D \ge 30$), full eigen-decomposition $O(D^3)$ is avoided by maintaining a truncated rank-$k$ spectral basis:
$$\mathbf{C}_{\mathcal{E}}^{(t)} \approx \mathbf{V}_k \mathbf{\Lambda}_k \mathbf{V}_k^\top + \sigma_{\text{res}}^2 \mathbf{I}_D, \quad k = \min(D, \max(5, \lfloor \sqrt{D} \rfloor))$$
Matrix operations (exponential, logarithm, geodesic interpolation, and Mahalanobis inverse) are evaluated in $O(k^2 D)$ time via the Woodbury matrix identity.

---

## 3. Online Landscape Ruggedness Sensor (LRS)

To eliminate manual tuning between exploration and exploitation, RMA-EA continuously senses the local topology:
1. **Directional Variation Coefficient ($V_i$)**:
   For random sampling probes along the principal geodesic axes:
   $$V_i = \frac{|f(\mathbf{x}_i + \delta \mathbf{u}_j) - f(\mathbf{x}_i)|}{\delta \cdot \sqrt{\lambda_j} \cdot (|f(\mathbf{x}_i)| + \epsilon_f)}$$
2. **Ensemble Ruggedness Index ($\rho_t \in [0, 1]$)**:
   Calculated as the normalized entropy of directional variations across the population:
   $$\rho_t = \operatorname{sigmoid}\left( \kappa \cdot \left(\frac{\operatorname{std}(V)}{\operatorname{mean}(V) + \epsilon} - \theta_{\text{base}}\right) \right)$$
   - $\rho_t \to 1$: Highly rugged / multimodal landscape $\implies$ expand search along transverse geodesic directions, increase exploration mutation factor.
   - $\rho_t \to 0$: Smooth quadratic / narrow ravine $\implies$ contract along manifold principal axes, accelerate exploitation.

---

## 4. Dual-Channel Adaptive Mutation & Crossover

For each target vector $\mathbf{x}_i$, candidate donor $\mathbf{v}_i$ is generated through one of two channels:

### Channel A: Geodesic Drift Exploration (Probability $P_{\text{expl}} = \rho_t$)
$$\mathbf{v}_i = \mathbf{x}_i + F_i (\mathbf{x}_{\text{pbest}} - \mathbf{x}_i) + F_i \cdot \sum_{j=1}^k \sqrt{\lambda_j} \xi_{i,j} \mathbf{v}_j + \zeta_i \cdot \mathcal{C}(0, \sigma_{\text{Cauchy}})$$
where $\mathcal{C}$ is a standard Cauchy perturbation directed orthogonally to avoid saddle traps.

### Channel B: Riemannian Anisotropic Contraction (Probability $P_{\text{expt}} = 1 - \rho_t$)
$$\mathbf{v}_i = \mathbf{x}_{\text{pbest}} + F_i (\mathbf{x}_{r_1} - \mathbf{x}_{r_2}^{\text{arch}}) + \eta_i \cdot \bar{\mathbf{C}}_t^{1/2} \boldsymbol{\mathcal{N}}(\mathbf{0}, \mathbf{I}_D)$$
where $\mathbf{x}_{r_2}^{\text{arch}}$ is sampled from the union of current population and external archive $\mathcal{A}$.

### Adaptive Memory Updating:
Historical memories $\mathbf{M}_F$ and $\mathbf{M}_{Cr}$ are maintained and updated using successful parameters through Lehmer mean:
$$mean_L(S_F) = \frac{\sum w_k F_k^2}{\sum w_k F_k}, \quad mean_A(S_{Cr}) = \sum w_k Cr_k$$

### Population Size Adaptation (LPSR):
$$N_{t+1} = \operatorname{round}\left(\frac{N_{\min} - N_{\text{init}}}{\text{MaxFES}} \cdot \text{FES} + N_{\text{init}}\right)$$

---

## 5. Verification and Acceptance Criteria
1. **Unit Tests**: Full unit test coverage for manifold operations, ruggedness sensing, operators, and benchmark functions (all passing).
2. **Benchmark Verification**: Tested on CEC2017 & CEC2022 functions across multiple dimensions (10D, 30D).
3. **Statistical Rigor**: Multi-run statistical evaluation (30 independent runs), Wilcoxon signed-rank tests ($p < 0.05$), Friedman rank ranking.
4. **Ablation Evidence**: Clear proof of the individual contributions of Riemannian manifold guidance and landscape sensing.
5. **Manuscript Readiness**: Submission-grade scientific paper draft following IEEE Transactions style.
