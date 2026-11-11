"""Fair, balanced and rotating team formation for project-based learning curricula."""
from .data import load_students, sample_cohort
from .algorithms import random_assignment, serpentine, anneal
from .metrics import module_metrics, repeat_share, add_pairs
from .curriculum import POLICIES, run_policy, run_cohort
from .api import form_teams, history_matrix

__all__ = ["load_students", "sample_cohort", "random_assignment", "serpentine", "anneal",
           "module_metrics", "repeat_share", "add_pairs", "POLICIES", "run_policy", "run_cohort", "form_teams", "history_matrix"]
__version__ = "1.0.0"
