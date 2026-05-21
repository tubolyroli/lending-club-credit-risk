"""Streamlit dashboard: predict default probability + expected profit + SHAP explanation.

Run with: uv run streamlit run dashboard/app.py
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from credit_risk import data as cr_data, models as cr_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# From notebook 04 — the calibrated PD threshold that maximized realized profit on test.
OPTIMAL_THRESHOLD = 0.2807

# Rough recovery assumption for charge-offs. Lending Club historically recovered
# ~30% of unpaid principal, so loss-given-default ≈ 70% of the loan amount.
LGD_FRACTION = 0.7


st.set_page_config(page_title="Credit Risk Decision Support", layout="wide")


@st.cache_resource
def load_model_bundle():
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm.txt"))
    iso = joblib.load(MODELS_DIR / "isotonic_lgb.joblib")
    explainer = shap.TreeExplainer(booster)
    return booster, iso, explainer


@st.cache_data
def load_feature_defaults():
    """Medians / modes from training data — used to fill features the user doesn't enter."""
    train = pd.read_parquet(cr_data.PROCESSED_DIR / "train.parquet")
    meta = ["defaulted", "realized_profit", "issue_d"]
    X = train.drop(columns=meta)

    defaults = {}
    cat_options = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            s = X[col].astype(object)
            defaults[col] = str(s.mode().iloc[0])
            cat_options[col] = sorted(s.dropna().unique().tolist())
    return defaults, cat_options, list(X.columns)


booster, iso, explainer = load_model_bundle()
defaults, cat_options, feature_order = load_feature_defaults()


st.title("Should we approve this loan?")
st.caption(
    f"Calibrated default probability, expected profit, and the factors driving the decision. "
    f"Approval rule (from notebook 04): **calibrated PD ≤ {OPTIMAL_THRESHOLD:.1%}**."
)


def _index_or_zero(options, value):
    try:
        return options.index(value)
    except (ValueError, KeyError):
        return 0


with st.sidebar:
    st.header("Loan application")

    st.subheader("Loan terms")
    loan_amnt = st.number_input("Loan amount ($)", 1_000, 40_000, 15_000, step=500)
    term_months = st.radio("Term (months)", [36, 60], horizontal=True)
    int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 13.0, 0.25)
    purpose = st.selectbox(
        "Purpose",
        cat_options["purpose"],
        index=_index_or_zero(cat_options["purpose"], "debt_consolidation"),
    )

    st.subheader("Borrower")
    annual_inc = st.number_input("Annual income ($)", 10_000, 500_000, 65_000, step=1_000)
    dti = st.slider("Debt-to-income ratio", 0.0, 50.0, 18.0, 0.5)
    fico = st.slider("FICO score (range low)", 660, 845, 690)
    emp_length = st.slider("Employment length (years)", 0.0, 10.0, 5.0, 0.5)
    home_ownership = st.selectbox(
        "Home ownership",
        cat_options["home_ownership"],
        index=_index_or_zero(cat_options["home_ownership"], "MORTGAGE"),
    )
    verification = st.selectbox(
        "Income verification",
        cat_options["verification_status"],
        index=0,
    )


# Map the user's term choice (36 / 60) to the exact string in the training data
# (LC data uses " 36 months" with a leading space, so use the loaded options).
term_str = next((s for s in cat_options["term"] if str(term_months) in s), cat_options["term"][0])

# Amortized monthly installment for a fixed-rate loan
r = int_rate / 100 / 12
installment = loan_amnt * r / (1 - (1 + r) ** (-term_months)) if r > 0 else loan_amnt / term_months

# Build the full feature row by merging user inputs over the defaults
features = dict(defaults)
features.update({
    "loan_amnt": float(loan_amnt),
    "int_rate": float(int_rate),
    "installment": float(installment),
    "term": term_str,
    "purpose": purpose,
    "annual_inc": float(annual_inc),
    "dti": float(dti),
    "fico_range_low": float(fico),
    "fico_range_high": float(fico + 4),
    "emp_length_years": float(emp_length),
    "home_ownership": home_ownership,
    "verification_status": verification,
    "installment_to_income": float(installment * 12 / annual_inc) if annual_inc > 0 else np.nan,
})

X_one = pd.DataFrame([features])[feature_order]
for col in X_one.columns:
    if not pd.api.types.is_numeric_dtype(X_one[col]):
        X_one[col] = X_one[col].astype(object)
X_lgb = cr_models.prepare_for_lgb(X_one)

# Predict
p_raw = float(booster.predict(X_lgb)[0])
p_cal = float(iso.predict(np.array([p_raw]))[0])
approved = p_cal <= OPTIMAL_THRESHOLD

# Expected profit. Interest = total amortized payments minus principal.
expected_interest = installment * term_months - loan_amnt
expected_loss = loan_amnt * LGD_FRACTION
expected_profit = (1 - p_cal) * expected_interest - p_cal * expected_loss


col1, col2, col3 = st.columns(3)
col1.metric("Calibrated default probability", f"{p_cal:.1%}")
col2.metric(
    "Decision",
    "✓ Approve" if approved else "✗ Reject",
    delta=f"threshold {OPTIMAL_THRESHOLD:.1%}",
    delta_color="normal" if approved else "inverse",
)
col3.metric("Expected profit ($)", f"{expected_profit:,.0f}")

st.divider()

st.subheader("Why this decision?")
st.caption("Top features pushing this loan toward 'safe' (blue) or 'risky' (red). Bars sum to the log-odds shift from the population baseline.")

explanation = explainer(X_lgb)
fig = plt.figure(figsize=(9, 6))
shap.plots.waterfall(explanation[0], max_display=10, show=False)
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

with st.expander("Show full feature vector (user inputs + medians for un-asked fields)"):
    df_show = X_one.T.rename(columns={X_one.index[0]: "value"})
    st.dataframe(df_show, use_container_width=True)

with st.expander("How this model was built"):
    st.markdown(
        "- Trained on **691k matured loans** issued 2007-2015 (Lending Club).\n"
        "- LightGBM with isotonic calibration on a 2015 hold-out fold.\n"
        "- Approval threshold chosen by maximizing realized profit on a separate 2016 test set.\n"
        "- Test-set lift over approve-all baseline: **+10.9% ($6.3M on $57.9M)**."
    )
