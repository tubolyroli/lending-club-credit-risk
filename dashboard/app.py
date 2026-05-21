"""Streamlit dashboard: predict default probability + expected profit + SHAP explanation.

Run with: uv run streamlit run dashboard/app.py
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

from credit_risk import data as cr_data, models as cr_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# From notebook 04: the calibrated PD threshold that maximized realized profit on test.
OPTIMAL_THRESHOLD = 0.2807

# Rough recovery assumption for charge-offs. Lending Club historically recovered
# ~30% of unpaid principal, so loss-given-default ≈ 70% of the loan amount.
LGD_FRACTION = 0.7

ACCENT = "#22d3ee"
GOOD = "#10b981"
BAD = "#ef4444"
WARN = "#f59e0b"
MUTED = "#94a3b8"
PANEL_BG = "#111a2e"
PAGE_BG = "#0b1220"

FRIENDLY_NAMES = {
    "int_rate": "Interest rate",
    "sub_grade": "Lending Club sub-grade",
    "grade": "Lending Club grade",
    "dti": "Debt-to-income ratio",
    "annual_inc": "Annual income",
    "fico_range_low": "FICO score",
    "fico_range_high": "FICO score (upper)",
    "loan_amnt": "Loan amount",
    "term": "Loan term",
    "installment": "Monthly installment",
    "installment_to_income": "Installment / income",
    "emp_length_years": "Employment length",
    "purpose": "Loan purpose",
    "home_ownership": "Home ownership",
    "verification_status": "Income verification",
    "revol_util": "Revolving utilization",
    "revol_bal": "Revolving balance",
    "open_acc": "Open credit accounts",
    "total_acc": "Total credit accounts",
    "delinq_2yrs": "Delinquencies (2y)",
    "inq_last_6mths": "Recent credit inquiries",
    "pub_rec": "Public records",
    "addr_state": "Borrower state",
}


st.set_page_config(
    page_title="Credit Risk Decision Support",
    page_icon="🏦",
    layout="wide",
)


UI_FONT = '"Inter Tight", system-ui, -apple-system, "Segoe UI", sans-serif'
SERIF_FONT = '"Source Serif 4", "Source Serif Pro", Georgia, serif'
MONO_FONT = '"JetBrains Mono", "SF Mono", Menlo, Consolas, monospace'


# ---- Global CSS polish ----
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700;8..60,800&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
        font-family: {UI_FONT};
        font-feature-settings: "ss01", "ss02", "cv11";
    }}
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: {SERIF_FONT};
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    h1 {{ letter-spacing: -0.02em; }}
    code, pre, kbd, samp, tt, .stCode, [data-testid="stMetricValue"] {{
        font-family: {MONO_FONT};
        font-variant-numeric: tabular-nums;
    }}
    .stNumberInput input, .stTextInput input {{ font-family: {MONO_FONT}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .decision-card {{
        border-radius: 14px;
        padding: 22px 24px;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
        height: 100%;
    }}
    .decision-card .label {{
        font-family: {UI_FONT};
        font-size: 12px; color: #94a3b8;
        letter-spacing: 0.14em; text-transform: uppercase;
        margin-bottom: 6px;
    }}
    .decision-card .verdict {{
        font-family: {SERIF_FONT};
        font-size: 36px; font-weight: 700; line-height: 1.1;
        letter-spacing: -0.01em;
        margin: 2px 0 6px 0;
    }}
    .decision-card .verdict.numeric {{
        font-family: {MONO_FONT};
        font-variant-numeric: tabular-nums;
        font-weight: 700;
    }}
    .decision-card .sub {{ font-size: 13px; color: #cbd5e1; }}
    .driver-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 14px; margin: 6px 0;
        background: rgba(148, 163, 184, 0.07);
        border-radius: 10px;
        border-left: 4px solid var(--accent, #94a3b8);
    }}
    .driver-row .name {{ font-weight: 600; color: #e2e8f0; }}
    .driver-row .meta {{ font-size: 12px; color: #94a3b8; font-family: {MONO_FONT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Streamlit reuses the same DOM nodes across reruns, so CSS @keyframes never replay.
# This script watches the parent doc for text changes on the verdict/metric/card nodes
# and replays the animation via the Web Animations API.
# Injected through components.html(height=0) because st.markdown strips <script> tags.
components.html(
    """
    <script>
    (function () {
        const doc = window.parent.document;
        if (doc.__lcAnimSetup) return;
        doc.__lcAnimSetup = true;

        const popIn = {
            keyframes: [
                { opacity: 0, transform: 'translateY(14px) scale(0.82)', offset: 0 },
                { opacity: 1, transform: 'translateY(-2px) scale(1.06)', offset: 0.55 },
                { opacity: 1, transform: 'translateY(0) scale(1)', offset: 1 }
            ],
            options: { duration: 520, easing: 'cubic-bezier(0.2, 0.7, 0.2, 1.2)' }
        };
        const cardFlash = {
            keyframes: [
                { filter: 'brightness(1)', boxShadow: '0 0 0 0 rgba(34,211,238,0.0), 0 6px 24px rgba(0,0,0,0.35)', offset: 0 },
                { filter: 'brightness(1.75)', boxShadow: '0 0 0 6px rgba(34,211,238,0.45), 0 6px 24px rgba(0,0,0,0.35)', offset: 0.15 },
                { filter: 'brightness(1)', boxShadow: '0 0 0 0 rgba(34,211,238,0.0), 0 6px 24px rgba(0,0,0,0.35)', offset: 1 }
            ],
            options: { duration: 700, easing: 'cubic-bezier(0.4, 0, 0.2, 1)' }
        };

        const lastText = new WeakMap();
        const targets = [
            { sel: '.decision-card .verdict', conf: popIn },
            { sel: '[data-testid="stMetricValue"]', conf: popIn },
            { sel: '.decision-card', conf: cardFlash },
        ];

        function scan() {
            for (const { sel, conf } of targets) {
                doc.querySelectorAll(sel).forEach(el => {
                    const text = el.textContent;
                    if (lastText.get(el) === text) return;
                    lastText.set(el, text);
                    el.animate(conf.keyframes, conf.options);
                });
            }
        }

        let timer = null;
        const obs = new MutationObserver(() => {
            clearTimeout(timer);
            timer = setTimeout(scan, 40);
        });
        obs.observe(doc.body, { childList: true, subtree: true, characterData: true });

        scan();
    })();
    </script>
    """,
    height=0,
)


@st.cache_resource
def load_model_bundle():
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm.txt"))
    iso = joblib.load(MODELS_DIR / "isotonic_lgb.joblib")
    explainer = shap.TreeExplainer(booster)
    return booster, iso, explainer


@st.cache_data
def load_feature_defaults():
    """Medians / modes from training data, used to fill features the user doesn't enter."""
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


# ---- Header ----
st.markdown(
    f"""
    <div style="margin-bottom: 8px;">
        <div style="color: {ACCENT}; font-size: 13px; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600;">
            Credit Risk &middot; Decision Support
        </div>
        <h1 style="margin: 4px 0 4px 0; font-weight: 800;">Should we approve this loan?</h1>
        <div style="color: {MUTED}; font-size: 15px;">
            Calibrated default probability, expected profit, and the factors driving the decision.
            Approve when calibrated PD &le; <strong style="color: {ACCENT}">{OPTIMAL_THRESHOLD:.1%}</strong>
            (profit-optimal threshold from notebook&nbsp;04).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ---- Sidebar inputs ----
def _index_or_zero(options, value):
    try:
        return options.index(value)
    except (ValueError, KeyError):
        return 0


with st.sidebar:
    st.markdown(
        f"<div style='color:{ACCENT}; font-size:11px; letter-spacing:0.16em; "
        f"text-transform:uppercase; font-weight:700;'>Loan application</div>",
        unsafe_allow_html=True,
    )
    st.markdown("##### Loan terms")
    loan_amnt = st.number_input("Loan amount ($)", 1_000, 40_000, 15_000, step=500)
    term_months = st.radio("Term (months)", [36, 60], horizontal=True)
    int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 13.0, 0.25)
    purpose = st.selectbox(
        "Purpose",
        cat_options["purpose"],
        index=_index_or_zero(cat_options["purpose"], "debt_consolidation"),
    )

    st.markdown("##### Borrower")
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


# LC encodes term as " 36 months" / " 60 months"; pick the matching string.
term_str = next((s for s in cat_options["term"] if str(term_months) in s), cat_options["term"][0])

# Amortized monthly installment for a fixed-rate loan
r = int_rate / 100 / 12
installment = loan_amnt * r / (1 - (1 + r) ** (-term_months)) if r > 0 else loan_amnt / term_months

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

p_raw = float(booster.predict(X_lgb)[0])
p_cal = float(iso.predict(np.array([p_raw]))[0])
approved = p_cal <= OPTIMAL_THRESHOLD

expected_interest = installment * term_months - loan_amnt
expected_loss = loan_amnt * LGD_FRACTION
weighted_interest = (1 - p_cal) * expected_interest
weighted_loss = p_cal * expected_loss
expected_profit = weighted_interest - weighted_loss


# ---- Hero row: gauge | decision | profit ----
def make_gauge(pd_pct: float, threshold_pct: float) -> go.Figure:
    danger = pd_pct > threshold_pct
    bar_color = BAD if danger else GOOD
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_pct,
        number={"suffix": "%", "font": {"size": 38, "color": "#e2e8f0", "family": "JetBrains Mono, monospace"}},
        gauge={
            "axis": {
                "range": [0, 50],
                "tickwidth": 1,
                "tickcolor": "#475569",
                "tickfont": {"color": "#94a3b8", "size": 11, "family": "Inter Tight, sans-serif"},
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold_pct], "color": "rgba(16,185,129,0.18)"},
                {"range": [threshold_pct, 50], "color": "rgba(239,68,68,0.18)"},
            ],
            "threshold": {
                "line": {"color": WARN, "width": 3},
                "thickness": 0.85,
                "value": threshold_pct,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=230,
        margin=dict(l=20, r=20, t=24, b=10),
    )
    return fig


hero_left, hero_mid, hero_right = st.columns([1.1, 1.1, 1.3])

with hero_left:
    st.markdown(
        f"<div class='label' style='color:{MUTED}; font-size:12px; letter-spacing:0.14em; "
        f"text-transform:uppercase; margin-bottom:-12px; padding-left:6px;'>"
        f"Default probability (calibrated)</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_gauge(p_cal * 100, OPTIMAL_THRESHOLD * 100), use_container_width=True, config={"displayModeBar": False})

with hero_mid:
    if approved:
        verdict, verdict_color = "Approve", GOOD
        bg_from, bg_to = "rgba(16,185,129,0.18)", "rgba(16,185,129,0.04)"
        sub = f"PD {p_cal:.1%} &le; threshold {OPTIMAL_THRESHOLD:.1%}"
    else:
        verdict, verdict_color = "Reject", BAD
        bg_from, bg_to = "rgba(239,68,68,0.18)", "rgba(239,68,68,0.04)"
        sub = f"PD {p_cal:.1%} &gt; threshold {OPTIMAL_THRESHOLD:.1%}"
    st.markdown(
        f"""
        <div class="decision-card" style="background: linear-gradient(135deg, {bg_from}, {bg_to});">
            <div class="label">Decision</div>
            <div class="verdict" style="color: {verdict_color};">{verdict}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    profit_color = GOOD if expected_profit >= 0 else BAD
    st.markdown(
        f"""
        <div class="decision-card" style="background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(34,211,238,0.02));">
            <div class="label">Expected profit (per loan)</div>
            <div class="verdict numeric" style="color: {profit_color};">${expected_profit:,.0f}</div>
            <div class="sub">
                +${weighted_interest:,.0f} interest weighted by (1&minus;PD)
                &nbsp;&minus;&nbsp;
                ${weighted_loss:,.0f} loss weighted by PD
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")
st.write("")

# ---- Tabs ----
tab_decision, tab_explain, tab_method = st.tabs(["Decision details", "Explainability", "Methodology"])


def make_profit_breakdown(weighted_interest: float, weighted_loss: float, net: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[weighted_interest, -weighted_loss, net],
        y=["Expected interest<br>(1 − PD) × interest_if_paid",
           "Expected loss<br>PD × loss_if_default",
           "<b>Net expected profit</b>"],
        orientation="h",
        marker_color=[GOOD, BAD, ACCENT if net >= 0 else BAD],
        text=[f"+${weighted_interest:,.0f}", f"−${weighted_loss:,.0f}", f"${net:,.0f}"],
        textposition="outside",
        textfont={"color": "#e2e8f0", "size": 13, "family": "JetBrains Mono, monospace"},
        hoverinfo="skip",
        width=0.55,
    ))
    abs_max = max(weighted_interest, weighted_loss, abs(net)) * 1.45
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=40, t=10, b=20),
        showlegend=False,
        xaxis=dict(
            range=[-abs_max, abs_max],
            zeroline=True, zerolinecolor="#475569", zerolinewidth=1,
            showgrid=False, showticklabels=False,
        ),
        yaxis=dict(showgrid=False, tickfont={"color": "#cbd5e1", "size": 12, "family": "Inter Tight, sans-serif"}),
    )
    return fig


with tab_decision:
    st.markdown("##### How the expected profit breaks down")
    st.caption(
        "If the loan is fully repaid, you earn interest. If it defaults, you lose ~70% of principal "
        "(historical Lending Club recovery). Weighting each by the calibrated default probability gives the net."
    )
    st.plotly_chart(make_profit_breakdown(weighted_interest, weighted_loss, expected_profit),
                    use_container_width=True, config={"displayModeBar": False})

    cols = st.columns(4)
    cols[0].metric("Calibrated PD", f"{p_cal:.1%}")
    cols[1].metric("Monthly installment", f"${installment:,.0f}")
    cols[2].metric("Interest if paid", f"${expected_interest:,.0f}")
    cols[3].metric("Loss if default", f"${expected_loss:,.0f}")

    with st.expander("Full feature vector (user inputs + medians for un-asked fields)"):
        # Cast to str so Arrow doesn't choke on the mixed-dtype transposed column.
        df_show = X_one.T.rename(columns={X_one.index[0]: "value"}).astype(str)
        st.dataframe(df_show, use_container_width=True)


def top_risk_drivers(explanation, k: int = 3):
    """Top-k features by |SHAP value| with friendly names, direction, and the raw input value."""
    shap_vals = explanation.values[0]
    feature_names = list(explanation.feature_names)
    feature_inputs = explanation.data[0]
    order = np.argsort(np.abs(shap_vals))[::-1][:k]
    out = []
    for idx in order:
        raw_name = feature_names[idx]
        val = shap_vals[idx]
        input_val = feature_inputs[idx]
        out.append({
            "name": FRIENDLY_NAMES.get(raw_name, raw_name),
            "raw_name": raw_name,
            "shap": float(val),
            "input": input_val,
            "direction": "risk" if val > 0 else "safe",
        })
    return out


def format_input(raw_name: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    if isinstance(value, (int, float)):
        if raw_name in {"annual_inc", "loan_amnt", "installment", "revol_bal"}:
            return f"${value:,.0f}"
        if raw_name in {"int_rate", "revol_util", "dti"}:
            return f"{value:.1f}"
        if raw_name == "installment_to_income":
            return f"{value:.2f}"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


with tab_explain:
    explanation = explainer(X_lgb)

    st.markdown("##### Top factors driving this decision")
    drivers = top_risk_drivers(explanation, k=3)
    for d in drivers:
        color = BAD if d["direction"] == "risk" else GOOD
        arrow = "▲ pushes toward default" if d["direction"] == "risk" else "▼ pushes toward safe"
        st.markdown(
            f"""
            <div class="driver-row" style="--accent: {color};">
                <div>
                    <div class="name">{d['name']}</div>
                    <div class="meta">Input: <code>{format_input(d['raw_name'], d['input'])}</code></div>
                </div>
                <div style="text-align: right;">
                    <div style="color: {color}; font-weight: 700;">{arrow}</div>
                    <div class="meta">SHAP {d['shap']:+.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### Full SHAP waterfall")
    st.caption(
        "Each bar is a feature's contribution to log-odds of default, summed from the population baseline "
        f"E[f(x)] to this loan's prediction f(x). Red increases risk, blue reduces it."
    )

    # Let shap.plots.waterfall manage its own figure (it calls plt.gcf() and
    # resizes internally). Pre-creating one here used to confuse its tick-label
    # bookkeeping for edge-case SHAP distributions and raised IndexError.
    try:
        with plt.style.context("dark_background"):
            shap.plots.waterfall(explanation[0], max_display=10, show=False)
            fig = plt.gcf()
            fig.patch.set_facecolor(PANEL_BG)
            for ax in fig.axes:
                ax.set_facecolor(PANEL_BG)
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                ax.tick_params(colors="#cbd5e1")
            st.pyplot(fig, clear_figure=True)
    except IndexError:
        plt.close("all")
        st.info(
            "Couldn't render the SHAP waterfall for this exact input combination "
            "(known shap+matplotlib edge case). The top drivers above are still correct."
        )


with tab_method:
    st.markdown(
        f"""
##### How this model was built

- **Training data.** ~691k matured Lending Club loans issued 2007–2015 (charged-off or fully paid before the snapshot).
- **Model.** LightGBM gradient boosting on native categoricals + isotonic calibration on a 2015 hold-out fold.
- **Threshold.** Approval cutoff of {OPTIMAL_THRESHOLD:.1%} chosen by maximizing realized profit on the 2016 test set,
  *not* by minimizing log-loss or maximizing AUC. Calibration matters because the expected-profit math needs
  honest probabilities.
- **Test-set lift.** +10.9% realized profit over the FICO≥660 approve-all baseline (+\$6.3M on \$57.9M).
- **Why this beats classification metrics.** A model that wins on AUC can lose on profit if its score
  distribution is wrong in the high-stakes tail. Profit curves at varying approval rates make the comparison directly.

##### Why values sometimes look "stuck" across slider changes

The calibrator is an **isotonic regression**, which is a step function. Small movements in the raw
LightGBM probability fall onto the same plateau and get mapped to an identical calibrated PD, so the
displayed PD and profit don't move even though the underlying score did. This is real model behavior,
not a UI bug. The SHAP waterfall (raw-score space) reflects the actual feature contributions in those cases.

##### Why these numbers, not others

- **Interest if paid:** sum of amortized monthly installments minus principal. Assumes the borrower pays on schedule.
- **Loss if default:** 70% of loan amount (Lending Club historically recovered ~30% of unpaid principal).
- **Expected profit:** `(1 − PD) × interest_if_paid − PD × loss_if_default`.

For the full analysis, see the [notebooks](https://github.com/tubolyroli/lending-club-credit-risk/tree/main/notebooks)
or the [README](https://github.com/tubolyroli/lending-club-credit-risk).
        """
    )
