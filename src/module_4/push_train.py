"""
Milestone 2: Fit/train handler for push notification purchase prediction.

Loads data, trains the model with given parametrisation, and saves to disk.
Designed to be invoked by an API (e.g. handler_fit(event, _)).
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
CRITICAL_COLUMNS = ["order_id", "user_id", "outcome", "product_type"]
REQUIRED_COLUMNS = [
    "variant_id",
    "product_type",
    "order_id",
    "user_id",
    "created_at",
    "order_date",
    "user_order_seq",
    "outcome",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
    "normalised_price",
    "discount_pct",
    "vendor",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]
# Preprocessing (dropped for multicollinearity)
DROP_COLUMNS = [
    "people_ex_baby",
    "std_days_to_buy_variant_id",
    "std_days_to_buy_product_type",
]
# Features: exclude ids, target, raw categoricals (we use _enc versions)
EXCLUDE_FEATURE_COLS = [
    "variant_id",
    "product_type",
    "order_id",
    "user_id",
    "created_at",
    "order_date",
    "outcome",
    "vendor",
]
# Dtype conversions
BOOLEAN_COLUMNS = [
    "outcome",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
]
INTEGER_COLUMNS = [
    "days_since_purchase_variant_id",
    "days_since_purchase_product_type",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
]

# Default model parametrisation (can be overridden via event)
DEFAULT_MODEL_PARAMETRISATION = {
    "model_type": "logistic_regression",
    "penalty": "l1",
    "C": 1.0,
    "solver": "saga",
    "max_iter": 1000,
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load training data from the given path."""
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows", len(df))
    return df


def validate_data(
    df: pd.DataFrame,
    on_null: Literal["raise", "drop"] = "raise",
    null_threshold: float = 0.10,
) -> pd.DataFrame:
    """Validate data schema and critical columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[CRITICAL_COLUMNS].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) == 0:
        return df

    n_total = len(df)
    rows_with_any_null = df[CRITICAL_COLUMNS].isna().any(axis=1).sum()
    null_frac = rows_with_any_null / n_total if n_total > 0 else 0.0

    if on_null == "raise":
        raise ValueError(
            f"Critical columns have nulls: {cols_with_nulls.to_dict()}. "
            "Use on_null='drop' to remove rows with nulls."
        )

    if null_frac > null_threshold:
        raise ValueError(
            f"Too many nulls ({null_frac:.1%} of rows). "
            f"Nulls by column: {cols_with_nulls.to_dict()}. "
            f"Threshold: {null_threshold:.1%}."
        )

    df_clean = df.dropna(subset=CRITICAL_COLUMNS)
    logger.warning(
        "Dropped rows with nulls in critical columns: %s", cols_with_nulls.to_dict()
    )
    return df_clean


def _build_model(config: dict) -> Pipeline:
    """Build a sklearn Pipeline (scaler + classifier) from config."""
    model_type = config.get("model_type", "logistic_regression").lower()

    if model_type == "logistic_regression":
        clf = LogisticRegression(
            random_state=RANDOM_STATE,
            penalty=config.get("penalty", "l1"),
            C=config.get("C", 1.0),
            solver=config.get("solver", "saga"),
            max_iter=config.get("max_iter", 1000),
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 10),
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            n_estimators=config.get("n_estimators", 100),
            learning_rate=config.get("learning_rate", 0.1),
            max_depth=config.get("max_depth", 5),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def preprocess(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    Preprocess data: filter orders with ≥5 items, time-based split, encode, scale.

    Time-based split: train 70%, validation 15%, test 15% of cumulative order count.
    """
    logger.info("Preprocessing data: filter, split, encode")
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["order_date"] = pd.to_datetime(df["order_date"])

    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")
    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    # Cap discount_pct at 1
    df["discount_pct"] = df["discount_pct"].apply(lambda x: 1 if x > 1 else x)

    # Drop columns with high multicollinearity (VIF > 10).
    # people_ex_baby = count_adults + count_children; std_* are highly correlated with avg_*.
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Filter: orders with at least 5 purchased items
    purchased_per_order = df.groupby("order_id")["outcome"].sum()
    valid_order_ids = purchased_per_order[purchased_per_order >= 5].index
    df = df[df["order_id"].isin(valid_order_ids)]
    logger.info("Filtered to %d rows (orders with ≥5 items)", len(df))

    # Time-based split (cumulative order count) to avoid data leakage
    order_to_date = df.groupby("order_id")["order_date"].min()
    order_dates_sorted = order_to_date.sort_values()
    n_orders = len(order_dates_sorted)
    train_cutoff = int(0.70 * n_orders)
    val_cutoff = int(0.85 * n_orders)

    train_order_ids = set(order_dates_sorted.index[:train_cutoff])
    val_order_ids = set(order_dates_sorted.index[train_cutoff:val_cutoff])
    test_order_ids = set(order_dates_sorted.index[val_cutoff:])

    df_train = df[df["order_id"].isin(train_order_ids)].copy()
    df_val = df[df["order_id"].isin(val_order_ids)].copy()
    df_test = df[df["order_id"].isin(test_order_ids)].copy()

    # Target encode (fit on train only to avoid data leakage)
    encoder_product = TargetEncoder(
        categories="auto",
        target_type="continuous",
        smooth="auto",
        cv=5,
        random_state=RANDOM_STATE,
    )
    encoder_vendor = TargetEncoder(
        categories="auto",
        target_type="continuous",
        smooth="auto",
        cv=5,
        random_state=RANDOM_STATE,
    )
    df_train["product_type_enc"] = encoder_product.fit_transform(
        df_train[["product_type"]], df_train["outcome"]
    )
    df_val["product_type_enc"] = encoder_product.transform(df_val[["product_type"]])
    df_test["product_type_enc"] = encoder_product.transform(df_test[["product_type"]])

    df_train["vendor_enc"] = encoder_vendor.fit_transform(
        df_train[["vendor"]], df_train["outcome"]
    )
    df_val["vendor_enc"] = encoder_vendor.transform(df_val[["vendor"]])
    df_test["vendor_enc"] = encoder_vendor.transform(df_test[["vendor"]])

    # Features (exclude ids, target, raw categoricals; we use _enc versions)
    exclude_cols = [c for c in EXCLUDE_FEATURE_COLS if c in df_train.columns]
    feature_cols = [
        c
        for c in df_train.columns
        if c not in exclude_cols and df_train[c].dtype in ["int64", "float64"]
    ]

    X_train = df_train[feature_cols].values
    y_train = df_train["outcome"].values
    X_val = df_val[feature_cols].values
    y_val = df_val["outcome"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["outcome"].values

    artifacts = {
        "encoder_product": encoder_product,
        "encoder_vendor": encoder_vendor,
        "feature_cols": feature_cols,
    }

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        artifacts,
    )


def train_and_save(
    data_path: str | Path,
    output_dir: str | Path,
    model_parametrisation: dict | None = None,
    on_null: Literal["raise", "drop"] = "raise",
) -> dict[str, str]:
    """
    Load data, train model, save pipeline.

    Args:
        data_path: Path to feature_frame CSV.
        output_dir: Directory to save the model artifacts.
        model_parametrisation: Model config (model_type, penalty, C, etc.).
            Defaults to logistic_regression with L1.
        on_null: How to handle nulls in critical columns.

    Returns:
        Dict with model_path and model_name (e.g. push_2026_03_14).
    """
    config = model_parametrisation or DEFAULT_MODEL_PARAMETRISATION
    model_name = f"push_{datetime.now().strftime('%Y_%m_%d')}"
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training model %s with config: %s", model_name, config)

    df = load_data(data_path)
    df = validate_data(df, on_null=on_null)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        artifacts,
    ) = preprocess(df)

    model = _build_model(config)
    logger.info("Fitting model on %d train samples", len(y_train))
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_precision = float(precision_score(y_val, y_val_pred, zero_division=0))
    val_recall = float(recall_score(y_val, y_val_pred, zero_division=0))
    logger.info(
        "Validation metrics: precision=%.4f, recall=%.4f", val_precision, val_recall
    )

    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "feature_cols": artifacts["feature_cols"],
        "model_parametrisation": config,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
    }

    # Save artifacts
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "encoder_product.pkl", "wb") as f:
        pickle.dump(artifacts["encoder_product"], f)
    with open(output_dir / "encoder_vendor.pkl", "wb") as f:
        pickle.dump(artifacts["encoder_vendor"], f)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    model_path = str(output_dir.resolve())
    logger.info("Model saved to %s", model_path)
    return {"model_path": model_path, "model_name": model_name}


def handler_fit(event: dict, _: Any = None) -> dict:
    """
    API handler for model training.

    Expected event keys:
        - model_parametrisation (optional): Dict with model_type, penalty, C, etc.
        - data_path (optional): Path to CSV. Default: module_4_datasets/feature_frame.csv
        - output_base (optional): Base dir for saving. Default: ./models
        - on_null (optional): "raise" or "drop". Default: "raise"

    Returns:
        Dict with statusCode and body (JSON string) containing model_path and model_name.
    """
    model_parametrisation = event.get(
        "model_parametrisation", DEFAULT_MODEL_PARAMETRISATION
    )
    script_dir = Path(__file__).resolve().parent
    data_path = event.get(
        "data_path", str(script_dir / "module_4_datasets" / "feature_frame.csv")
    )
    output_base = event.get("output_base", str(script_dir / "models"))
    on_null = event.get("on_null", "raise")

    result = train_and_save(
        data_path=data_path,
        output_dir=output_base,
        model_parametrisation=model_parametrisation,
        on_null=on_null,
    )

    return {
        "statusCode": "200",
        "body": json.dumps(result),
    }


if __name__ == "__main__":
    event = {
        "model_parametrisation": {
            "model_type": "random_forest",
            "n_estimators": 50,
            "max_depth": 5,
        }
    }
    out = handler_fit(event)
    print(out["body"])
