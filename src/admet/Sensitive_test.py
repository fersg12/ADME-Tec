import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from scipy.stats import spearmanr

# Import desirability routines from the local package
from .desirability_score import (
    normalize_weights,
    compute_desirability,
    compute_desirability_geometric,
)


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """Compute Spearman correlation robustly; return 0 if undefined."""
    try:
        rho, _ = spearmanr(a, b)
        if np.isnan(rho):
            return 0.0
        return float(rho)
    except Exception:
        return 0.0


def perturb_weights_sensitivity(weights, perturbation=0.1):
    import pandas as pd

    perturbed_sets = []
    debug_rows = []

    if not weights:
        return [], pd.DataFrame()

    total = sum(weights.values())
    base = {k: float(v) / total for k, v in weights.items()}

    def adjusted_factor(w: float, delta: float) -> float:
        denom = (1 - w * (1 + delta))
        if denom == 0:
            return 1.0
        return ((1 + delta) * (1 - w)) / denom

    for key in list(base.keys()):
        w = base[key]

        for delta in (-abs(perturbation), abs(perturbation)):
            f = adjusted_factor(w, delta)
            new_w = base.copy()
            new_w[key] = max(0.0, new_w[key] * f)

            tot2 = sum(new_w.values())
            new_w = {k: v / tot2 for k, v in new_w.items()}

            perturbed_sets.append((key, delta, new_w))

            row = {"property": key, "delta": delta, "original_weight": w}
            row.update(new_w)
            debug_rows.append(row)

    return perturbed_sets, pd.DataFrame(debug_rows)
