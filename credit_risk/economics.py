"""Economic model: turn predicted default probabilities into approval decisions and profit."""

import numpy as np
import pandas as pd


def expected_profit_per_loan(
    proba_default: np.ndarray,
    interest_if_paid: np.ndarray,
    loss_if_default: np.ndarray,
) -> np.ndarray:
    """E[profit] = (1 - p) * interest_if_paid - p * loss_if_default."""
    return (1 - proba_default) * interest_if_paid - proba_default * loss_if_default


def realized_profit(
    approved: np.ndarray,
    outcomes: np.ndarray,
    interest_if_paid: np.ndarray,
    loss_if_default: np.ndarray,
) -> float:
    """Sum of realized profit on the approved set, using actual outcomes (0/1 default)."""
    paid_mask = approved & (outcomes == 0)
    default_mask = approved & (outcomes == 1)
    return float(interest_if_paid[paid_mask].sum() - loss_if_default[default_mask].sum())


def profit_curve(
    scores: np.ndarray,
    realized_profit: np.ndarray,
    outcomes: np.ndarray | None = None,
    n_points: int = 100,
) -> pd.DataFrame:
    """Cumulative realized profit when approving the top-k safest loans by `scores`.

    `scores` is treated such that *lower is safer* (e.g. predicted default
    probability). Pass `-fico_range_low` to use FICO as a baseline.

    The curve sweeps approval rates from ~1% up to 100%; the value at
    approval_rate=1.0 equals the approve-all profit.
    """
    scores = np.asarray(scores)
    realized = np.asarray(realized_profit)
    order = np.argsort(scores, kind="stable")  # ascending — safest first
    cum_profit = np.cumsum(realized[order])
    n = len(scores)

    ks = np.unique(np.linspace(1, n, n_points).astype(int))
    rows = []
    if outcomes is not None:
        outcomes_sorted = np.asarray(outcomes)[order]
        cum_defaults = np.cumsum(outcomes_sorted)
    for k in ks:
        row = {
            "approval_rate": k / n,
            "n_approved": int(k),
            "score_threshold": float(scores[order[k - 1]]),
            "realized_profit": float(cum_profit[k - 1]),
            "profit_per_loan": float(cum_profit[k - 1] / k),
        }
        if outcomes is not None:
            row["default_rate"] = float(cum_defaults[k - 1] / k)
        rows.append(row)
    return pd.DataFrame(rows)
