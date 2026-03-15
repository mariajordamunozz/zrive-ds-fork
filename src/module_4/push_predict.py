"""
Milestone 2: Inference handler for push notification purchase prediction.

Loads a trained model and returns predictions for the given users.
Designed to be invoked by an API (e.g. handler_predict(event, _)).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path) -> tuple[Any, Any, Any, dict]:
    """
    Load model pipeline and encoders from disk.

    Args:
        model_path: Directory containing model.pkl, encoder_product.pkl,
            encoder_vendor.pkl, metadata.json.

    Returns:
        Tuple of (model, encoder_product, encoder_vendor, metadata).
    """
    model_path = Path(model_path)
    logger.info("Loading model from %s", model_path)
    with open(model_path / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_path / "encoder_product.pkl", "rb") as f:
        encoder_product = pickle.load(f)
    with open(model_path / "encoder_vendor.pkl", "rb") as f:
        encoder_vendor = pickle.load(f)
    with open(model_path / "metadata.json") as f:
        metadata = json.load(f)
    logger.info("Model loaded successfully")
    return model, encoder_product, encoder_vendor, metadata


def predict(
    users: dict[str, dict[str, Any]],
    model_path: str | Path,
) -> dict[str, float | int]:
    """
    Run inference on user-product pairs.

    Args:
        users: Dict mapping user_id (or row_id) to feature dict.
            Each feature dict must include product_type, vendor, and other
            required features.
        model_path: Path to the saved model directory.

    Returns:
        Dict mapping user_id to prediction (0 or 1).
    """
    model, encoder_product, encoder_vendor, metadata = load_model(model_path)
    logger.info("Running inference on %d samples", len(users))
    feature_cols = metadata["feature_cols"]

    # Convert to DataFrame (orient="index": keys are index, values are rows)
    df = pd.DataFrame.from_dict(users, orient="index")
    df.index = df.index.astype(str)  # Ensure string index for JSON compatibility

    # Target encode product_type and vendor
    if "product_type" in df.columns:
        df["product_type_enc"] = encoder_product.transform(df[["product_type"]])
    if "vendor" in df.columns:
        df["vendor_enc"] = encoder_vendor.transform(df[["vendor"]])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for inference: {missing}")

    X = df[feature_cols].values
    y_pred = model.predict(X)

    # Map index (user_id) to prediction
    predictions = {str(uid): int(p) for uid, p in zip(df.index, y_pred)}
    logger.info("Inference complete: %d predictions", len(predictions))
    return predictions


def handler_predict(event: dict, _: Any = None) -> dict:
    """
    API handler for inference.

    Expected event keys:
        - users: JSON string or dict. Structure:
            {user_id: {feature1: value, feature2: value, ...}, ...}
          Each row represents a user-product pair. Must include product_type,
          vendor, and all feature_cols.
        - model_path (optional): Path to saved model. If not provided, uses
          the most recent model in ./models/ (push_yyyy_mm_dd).

    Returns:
        Dict with statusCode and body (JSON string) containing prediction:
        {"prediction": {user_id: 0|1, ...}}
    """
    users_raw = event.get("users")
    if users_raw is None:
        logger.warning("Missing 'users' in event")
        return {
            "statusCode": "400",
            "body": json.dumps({"error": "Missing 'users' in event"}),
        }

    if isinstance(users_raw, str):
        users = json.loads(users_raw)
    else:
        users = users_raw

    script_dir = Path(__file__).resolve().parent
    model_path = event.get("model_path")
    if model_path is None:
        models_dir = script_dir / "models"
        if models_dir.exists():
            # Use most recent push_* directory
            push_dirs = sorted(
                models_dir.glob("push_*"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            model_path = push_dirs[0] if push_dirs else None
        if model_path is None:
            logger.error("No model found. Run push_train first or provide model_path.")
            return {
                "statusCode": "500",
                "body": json.dumps(
                    {
                        "error": "No model found. Run push_train first or provide model_path."
                    }
                ),
            }

    try:
        predictions = predict(users, model_path)
        return {
            "statusCode": "200",
            "body": json.dumps({"prediction": predictions}),
        }
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return {
            "statusCode": "500",
            "body": json.dumps({"error": str(e)}),
        }


if __name__ == "__main__":
    # Example: run with sample input
    sample_users = {
        "user_1": {
            "user_order_seq": 5,
            "ordered_before": 1,
            "abandoned_before": 0,
            "active_snoozed": 0,
            "set_as_regular": 0,
            "normalised_price": 0.5,
            "discount_pct": 0.1,
            "global_popularity": 0.02,
            "count_adults": 2,
            "count_children": 0,
            "count_babies": 0,
            "count_pets": 0,
            "people_ex_baby": 2,
            "days_since_purchase_variant_id": 30,
            "avg_days_to_buy_variant_id": 35,
            "days_since_purchase_product_type": 25,
            "avg_days_to_buy_product_type": 30,
            "product_type": "ricepastapulses",
            "vendor": "default_vendor",
        },
    }
    event = {"users": json.dumps(sample_users)}
    out = handler_predict(event)
    print(out["body"])
