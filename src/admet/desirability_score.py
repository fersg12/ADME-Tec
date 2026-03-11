import pandas as pd
import numpy as np

from .desirability_conf import PROPERTY_CONFIG, desirability_families


def normalize_weights(weights: dict, eps: float = 1e-6) -> dict:
    """
    Normalize a dictionary of weights so that they sum to 1.
    """
    total = sum(weights.values())

    if total < eps:
        raise ValueError("Sum of weights is zero. Cannot normalize.")

    return {k: v / total for k, v in weights.items()}



def resolve_params(param_map, ranges):
    resolved = {}
    for k, v in param_map.items():
        if isinstance(v, str):
            if "[" in v:
                key, idx = v.replace("]", "").split("[")
                resolved[k] = ranges[key][int(idx)]
            else:
                val = ranges[v]
                if isinstance(val, (list, tuple, np.ndarray)):
                    raise ValueError(f"Parameter '{v}' must be scalar")
                resolved[k] = val
        else:
            resolved[k] = v
    return resolved


# ============================================================
# Desirability functions (UNCHANGED logic)
# ============================================================

def compute_desirability(
    inputs: pd.DataFrame,
    ranges: dict,
    weights: dict,
    config: dict,
) -> pd.DataFrame:
    """
    Arithmetic (linear) desirability scoring.
    Assumes:
        - inputs already contains ADME columns
        - weights are normalized
        - config keys match column names
    """

    if inputs is None or inputs.empty:
        raise ValueError("Input DataFrame is empty.")

    df = inputs.copy()

    weighted_sum = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for prop, cfg in config.items():

        # Column must exist
        if prop not in df.columns:
            continue

        # Weight must exist
        weight_key = cfg.get("weight")
        if weight_key not in weights:
            continue

        fn = desirability_families[cfg["family"]]
        params = resolve_params(cfg["params"], ranges)

        d = fn(df[prop], **params)
        df[f"desirability_{prop}"] = d

        w = weights[weight_key]
        weighted_sum += d * w
        total_weight += w

    if total_weight == 0:
        raise ZeroDivisionError(
            "No valid properties were used in desirability calculation. "
            "Check column names and weight keys."
        )

    df["Desirability Score"] = weighted_sum / total_weight

    return df.sort_values("Desirability Score", ascending=False)


def compute_desirability_geometric(
    inputs: pd.DataFrame,
    ranges: dict,
    weights: dict,
    config: dict,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Geometric desirability scoring.
    Penalizes weak properties more strongly.
    """

    if inputs is None or inputs.empty:
        raise ValueError("Input DataFrame is empty.")

    df = inputs.copy()

    log_sum = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for prop, cfg in config.items():

        if prop not in df.columns:
            continue

        weight_key = cfg.get("weight")
        if weight_key not in weights:
            continue

        fn = desirability_families[cfg["family"]]
        params = resolve_params(cfg["params"], ranges)

        # Avoid log(0)
        d = np.clip(fn(df[prop], **params), eps, 1.0)
        df[f"desirability_{prop}"] = d

        w = weights[weight_key]
        log_sum += w * np.log(d)
        total_weight += w

    if total_weight == 0:
        raise ZeroDivisionError(
            "No valid properties were used in geometric desirability calculation."
        )

    df["Desirability Score"] = np.exp(log_sum / total_weight)

    return df.sort_values("Desirability Score", ascending=False)
