"""Model training and calibration."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_cols(X: pd.DataFrame) -> list[str]:
    """Numeric (including bool) columns. Uses pandas' own dtype check so it
    handles object, StringDtype, the new pandas 3.x `str` dtype, and
    PyArrow-backed strings uniformly."""
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def categorical_cols(X: pd.DataFrame) -> list[str]:
    """Anything not numeric. Datetime columns should be removed by the caller."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def train_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """LR with median-impute + scale + one-hot.

    `min_frequency=100` collapses rare addr_state / sub_grade levels into an
    "infrequent" bucket, which keeps the encoded feature count manageable and
    prevents overfit on tiny strata.
    """
    num_cols = numeric_cols(X)
    cat_cols = categorical_cols(X)

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=100,
            sparse_output=False,
        )),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe


def prepare_for_lgb(X: pd.DataFrame) -> pd.DataFrame:
    """Convert string/object columns to pandas Categorical for native LGBM handling."""
    X = X.copy()
    for c in categorical_cols(X):
        X[c] = X[c].astype("category")
    return X


def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    num_boost_round: int = 500,
    seed: int = 0,
) -> lgb.Booster:
    """LightGBM with sensible defaults for tabular credit data.

    When `eval_set` is provided, trains with early stopping (patience=20).
    """
    cat_cols = categorical_cols(X)
    X_lgb = prepare_for_lgb(X)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": seed,
        "verbose": -1,
    }

    train_set = lgb.Dataset(X_lgb, label=y, categorical_feature=cat_cols)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: list = []
    if eval_set is not None:
        X_val, y_val = eval_set
        val_set = lgb.Dataset(
            prepare_for_lgb(X_val),
            label=y_val,
            categorical_feature=cat_cols,
            reference=train_set,
        )
        valid_sets.append(val_set)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(20, verbose=False))

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Default probability for a fitted LR pipeline or LightGBM booster."""
    if isinstance(model, lgb.Booster):
        return model.predict(prepare_for_lgb(X))
    return model.predict_proba(X)[:, 1]


def calibrate(scores: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression mapping raw scores to calibrated probabilities."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y)
    return iso
