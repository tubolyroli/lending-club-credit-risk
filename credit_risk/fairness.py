"""Group-level fairness metrics for credit decisions."""

import numpy as np
import pandas as pd


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    approved_col: str = "approved",
    outcome_col: str = "defaulted",
    profit_col: str | None = "realized_profit",
    min_n: int = 100,
) -> pd.DataFrame:
    """Per-group: sample size, approval rate, default rate among approved, profit per approved loan.

    Groups smaller than `min_n` are dropped to avoid noisy ratios.
    """
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) < min_n:
            continue
        approved = sub[sub[approved_col]]
        rows.append({
            group_col: g,
            "n": len(sub),
            "n_approved": len(approved),
            "approval_rate": float(sub[approved_col].mean()),
            "default_rate_all": float(sub[outcome_col].mean()),
            "default_rate_approved": (
                float(approved[outcome_col].mean()) if len(approved) > 0 else np.nan
            ),
            "profit_per_approved": (
                float(approved[profit_col].mean())
                if profit_col is not None and len(approved) > 0
                else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values("approval_rate", ascending=False).reset_index(drop=True)


def disparate_impact(
    metrics: pd.DataFrame,
    group_col: str,
    reference_group,
) -> pd.DataFrame:
    """Apply the 80% rule: each group's approval rate / reference group's approval rate.

    The classic threshold is 0.8 (a ratio below that is the flag).
    """
    ref_rows = metrics[metrics[group_col] == reference_group]
    if ref_rows.empty:
        raise ValueError(f"Reference group {reference_group!r} not found in metrics")
    ref_rate = float(ref_rows["approval_rate"].iloc[0])
    out = metrics.copy()
    out["disparate_impact_ratio"] = (out["approval_rate"] / ref_rate).round(3)
    out["flagged"] = out["disparate_impact_ratio"] < 0.8
    return out
