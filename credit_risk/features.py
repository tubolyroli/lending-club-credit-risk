"""Feature engineering for the credit risk model."""

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "total_acc",
    "revol_bal",
    "revol_util",
    "collections_12_mths_ex_med",
    "mort_acc",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "acc_now_delinq",
]

CATEGORICAL_FEATURES = [
    "term",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "initial_list_status",
]


def _parse_emp_length(s) -> float:
    """Convert '10+ years' / '< 1 year' / '5 years' / NaN to float years."""
    if pd.isna(s):
        return float("nan")
    s = str(s).strip()
    if s == "< 1 year":
        return 0.0
    if s == "10+ years":
        return 10.0
    parts = s.split()
    return float(parts[0]) if parts and parts[0].isdigit() else float("nan")


def _parse_pct(series: pd.Series) -> pd.Series:
    """LC stores int_rate / revol_util as either 13.49 (numeric) or '13.49%' (string)."""
    if series.dtype == object:
        return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")
    return series


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready feature DataFrame.

    Feature columns only — no target, no leakage. Missingness is preserved;
    imputation is the model pipeline's job.
    """
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    out = df[available_num + available_cat].copy()

    for col in ("int_rate", "revol_util"):
        if col in out.columns:
            out[col] = _parse_pct(out[col])

    if "emp_length" in df.columns:
        out["emp_length_years"] = df["emp_length"].map(_parse_emp_length)

    if "annual_inc" in df.columns and "installment" in df.columns:
        # Use np.nan (not pd.NA) so the result stays float64; pd.NA propagates
        # a nullable Float64 dtype that breaks downstream pd.qcut / sklearn.
        annual = df["annual_inc"].replace(0, np.nan)
        out["installment_to_income"] = (df["installment"] * 12) / annual

    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        eclm = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
        out["credit_history_years"] = (df["issue_d"] - eclm).dt.days / 365.25

    return out


def target(df: pd.DataFrame) -> pd.Series:
    """Binary target: 1 if Charged Off, 0 if Fully Paid."""
    return (df["loan_status"] == "Charged Off").astype(int).rename("defaulted")
