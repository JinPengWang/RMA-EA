"""Statistical analysis package for evolutionary computation experiments."""
from analysis.statistics import (
    wilcoxon_test,
    friedman_test,
    summarize_run_statistics,
    compute_win_tie_loss
)

__all__ = [
    "wilcoxon_test",
    "friedman_test",
    "summarize_run_statistics",
    "compute_win_tie_loss"
]
