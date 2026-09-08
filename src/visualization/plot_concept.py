"""Generate Figure 1: Conceptual Architecture of RMA-EA.

Visualizes:
(a) SPD Manifold Geometry: Euclidean swelling vs Log-Euclidean Geodesic.
(b) Dual-Channel Architecture: Online Landscape Sensor driving Geodesic Drift vs Anisotropic Contraction.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


def generate_figure_1(output_dir: str = "paper/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "axes.linewidth": 1.0
    })
    
    fig = plt.figure(figsize=(11.0, 5.0), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3], wspace=0.25)
    
    # -------------------------------------------------------------
    # Panel (a): SPD Manifold Cone & Geodesic Interpolation
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    
    # Draw open cone of SPD matrices
    cone_x = np.linspace(0, 4.0, 100)
    cone_y_upper = 1.0 * cone_x
    cone_y_lower = -0.3 * cone_x
    
    ax1.fill_between(cone_x, cone_y_lower, cone_y_upper, color="#e6f2ff", alpha=0.6, label="SPD Manifold $\\mathcal{S}_{++}^D$")
    ax1.plot(cone_x, cone_y_upper, color="#1f77b4", linewidth=1.5)
    ax1.plot(cone_x, cone_y_lower, color="#1f77b4", linewidth=1.5)
    
    # Points C1 and C2 on the manifold
    p1 = (1.2, 0.7)
    p2 = (3.4, 0.4)
    
    # Euclidean line (straight chord - suffers from swelling effect)
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="#d62728", linewidth=1.8, label="Euclidean Interpolation (Swelling)")
    
    # Intrinsic Geodesic curve gamma(tau) = exp((1-tau)log(C1) + tau*log(C2))
    tau = np.linspace(0, 1, 50)
    # Parametric curve bowing naturally towards the cone interior
    geo_x = (1 - tau) * p1[0] + tau * p2[0]
    geo_y = (1 - tau) * p1[1] + tau * p2[1] + 0.35 * np.sin(np.pi * tau)
    ax1.plot(geo_x, geo_y, "-", color="#2ca02c", linewidth=2.4, label="Riemannian Geodesic $\\gamma(\\tau)$")
    
    # Plot covariance endpoints
    ax1.plot(p1[0], p1[1], "o", color="#1f77b4", markersize=8)
    ax1.text(p1[0] - 0.35, p1[1] + 0.1, "$\\mathbf{C}_1$", fontsize=12, fontweight="bold")
    
    ax1.plot(p2[0], p2[1], "o", color="#1f77b4", markersize=8)
    ax1.text(p2[0] + 0.1, p2[1] - 0.05, "$\\mathbf{C}_2$", fontsize=12, fontweight="bold")
    
    # Midpoint / Barycenter
    mid_idx = len(tau) // 2
    ax1.plot(geo_x[mid_idx], geo_y[mid_idx], "D", color="#ff7f0e", markersize=8)
    ax1.text(geo_x[mid_idx] - 0.1, geo_y[mid_idx] + 0.18, "Riemannian Barycenter $\\bar{\\mathbf{C}}_t$", fontsize=9.5, fontweight="bold", color="#ff7f0e")
    
    ax1.set_xlim(0, 4.2)
    ax1.set_ylim(-1.5, 3.8)
    ax1.set_title("(a) SPD Riemannian Manifold Geometry", loc="left", fontweight="bold")
    ax1.axis("off")
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ddd", fontsize=8.5)
    
    # -------------------------------------------------------------
    # Panel (b): Dual-Channel Search & Landscape Sensor Workflow
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("(b) Dual-Channel Geodesic Search Framework", loc="left", fontweight="bold")
    
    def draw_box(ax, xy, w, h, text, color="#f0f4f8", border="#1f77b4", bold=False):
        box = patches.FancyBboxPatch(
            xy, w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=border, linewidth=1.5
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
                fontsize=9.5, fontweight=weight, color="#111", multialignment="center")
                
    def draw_arrow(ax, start, end, label=""):
        arrow = patches.FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            color="#333", linewidth=1.4
        )
        ax.add_patch(arrow)
        if label:
            mid = ((start[0] + end[0])/2, (start[1] + end[1])/2)
            ax.text(mid[0], mid[1] + 0.25, label, ha="center", va="bottom", fontsize=8.5, color="#444")
            
    # Draw Boxes
    draw_box(ax2, (0.5, 7.5), 3.0, 1.4, "Population $\\mathcal{P}_t$\nElite Subgroup $\\mathcal{E}_t$", color="#e3f2fd", border="#1565c0", bold=True)
    draw_box(ax2, (5.5, 7.5), 4.0, 1.4, "SPD Manifold Covariance\n$\\mathbf{C}_{\\mathcal{E}}^{(t)} \\in \\mathcal{S}_{++}^D$ (Log-Euclidean)", color="#e8f5e9", border="#2e7d32", bold=True)
    draw_box(ax2, (3.0, 4.5), 4.2, 1.5, "Online Landscape Sensor (LRS)\nRuggedness Index $\\rho_t \\in [0, 1]$\nCurvature Roughness Fluctuation", color="#fff3e0", border="#e65100", bold=True)
    
    draw_box(ax2, (0.5, 1.2), 4.0, 1.8, "Channel A: Geodesic Drift\nExploration ($P_{\\text{expl}} = \\rho_t$)\n$\\bullet$ Non-Euclidean Geodesic Jump\n$\\bullet$ Orthogonal Cauchy Escape", color="#fce4ec", border="#c2185b")
    draw_box(ax2, (5.5, 1.2), 4.0, 1.8, "Channel B: Riemannian Contraction\nExploitation ($P_{\\text{expt}} = 1 - \\rho_t$)\n$\\bullet$ Covariance Ellipsoid Shrinkage\n$\\bullet$ Trust-Region Fine Descent", color="#ede7f6", border="#512da8")
    
    # Connect Arrows
    draw_arrow(ax2, (3.5, 8.2), (5.5, 8.2), label="Moments")
    draw_arrow(ax2, (7.5, 7.5), (6.5, 6.0), label="Decomposition")
    draw_arrow(ax2, (2.0, 7.5), (4.0, 6.0), label="Probing")
    draw_arrow(ax2, (4.5, 4.5), (2.5, 3.0), label="High $\\rho_t$ (Rugged)")
    draw_arrow(ax2, (6.0, 4.5), (7.5, 3.0), label="Low $\\rho_t$ (Smooth)")
    
    plt.tight_layout()
    png_path = os.path.join(output_dir, "fig1_concept_manifold.png")
    pdf_path = os.path.join(output_dir, "fig1_concept_manifold.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 1: {png_path} and {pdf_path}")


if __name__ == "__main__":
    generate_figure_1()
