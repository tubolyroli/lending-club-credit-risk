"""Loading and temporal splitting for Lending Club data."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

TRAIN_END_YEAR = 2015
TEST_START_YEAR = 2016

COMPLETED_STATUSES = {"Fully Paid", "Charged Off"}

# Columns that are only known after a loan is issued — must not be used as features.
# Kept in the DataFrame for computing realized profit on the test period.
POST_ISSUANCE_COLUMNS = [
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "out_prncp",
    "out_prncp_inv",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
]


def load_raw(filename: str = "accepted_2007_to_2018Q4.csv") -> pd.DataFrame:
    """Load the raw Lending Club CSV.

    Parses `issue_d` (e.g. "Dec-2015") into a real datetime so downstream code
    can group by year without re-parsing.
    """
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    return df


def filter_to_completed_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only loans with a final outcome (Fully Paid or Charged Off).

    Loans with status `Current`, `In Grace Period`, `Late ...` are still in flight —
    we don't yet know whether they'll default, so they can't be used for training or evaluation.
    """
    return df[df["loan_status"].isin(COMPLETED_STATUSES)].copy()


def temporal_split(
    df: pd.DataFrame, issue_date_col: str = "issue_d"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, test) on issue date — no random shuffling.

    Train: issued <= TRAIN_END_YEAR. Test: issued >= TEST_START_YEAR.
    """
    year = df[issue_date_col].dt.year
    train = df[year <= TRAIN_END_YEAR].copy()
    test = df[year >= TEST_START_YEAR].copy()
    return train, test


def filter_to_matured_loans(
    df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp = "2018-12-31",
) -> pd.DataFrame:
    """Keep only loans whose full term has elapsed by `snapshot_date`.

    Loans that finalize close to the dataset snapshot are a biased sample —
    early defaulters and early prepayers dominate, while loans with normal
    timelines are still in flight and don't appear as completed at all. Their
    realized economics look much worse than the underlying population's.
    Filtering by maturity removes this bias.

    `term` is parsed from the LC string format (e.g. " 36 months").
    """
    snapshot = pd.Timestamp(snapshot_date)
    term_months = df["term"].str.extract(r"(\d+)", expand=False).astype(int)
    expected_end = df["issue_d"] + pd.to_timedelta(term_months * 30, unit="D")
    return df[expected_end <= snapshot].copy()
