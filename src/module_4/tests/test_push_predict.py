"""Unit tests for push_predict.py."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from module_4 import push_train, push_predict

# Sample users dict for prediction.
SAMPLE_USERS = {
    "user_1": {
        "user_order_seq": 1,
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
        "days_since_purchase_variant_id": 30,
        "avg_days_to_buy_variant_id": 35,
        "days_since_purchase_product_type": 25,
        "avg_days_to_buy_product_type": 30,
        "product_type": "ricepastapulses",
        "vendor": "test_vendor",
    },
}


def _make_sample_feature_frame(tmp_path):
    """Create a minimal valid feature_frame CSV for testing."""
    rows = []
    for order_id in range(1, 21):
        has_5_items = order_id <= 10
        for i in range(10):
            outcome = 1.0 if (has_5_items and i < 5) else 0.0
            rows.append(
                {
                    "variant_id": 1000 + order_id,
                    "product_type": "ricepastapulses",
                    "order_id": order_id,
                    "user_id": order_id * 100,
                    "created_at": f"2020-10-{order_id:02d} 12:00:00",
                    "order_date": f"2020-10-{order_id:02d} 00:00:00",
                    "user_order_seq": 1,
                    "outcome": outcome,
                    "ordered_before": 0.0,
                    "abandoned_before": 0.0,
                    "active_snoozed": 0.0,
                    "set_as_regular": 0.0,
                    "normalised_price": 0.5,
                    "discount_pct": 0.1,
                    "vendor": "test_vendor",
                    "global_popularity": 0.02,
                    "count_adults": 2,
                    "count_children": 0,
                    "count_babies": 0,
                    "count_pets": 0,
                    "people_ex_baby": 2,
                    "days_since_purchase_variant_id": 30,
                    "avg_days_to_buy_variant_id": 35,
                    "std_days_to_buy_variant_id": 5.0,
                    "days_since_purchase_product_type": 25,
                    "avg_days_to_buy_product_type": 30,
                    "std_days_to_buy_product_type": 4.0,
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "feature_frame.csv"
    df.to_csv(path, index=False)
    return path


def _get_trained_model_dir(tmp_path):
    path = _make_sample_feature_frame(tmp_path)
    result = push_train.train_and_save(
        data_path=path,
        output_dir=tmp_path,
        model_parametrisation={"model_type": "logistic_regression", "max_iter": 500},
    )
    return Path(result["model_path"])


class TestLoadModel:
    def test_load_model_returns_tuple(self, tmp_path):
        model_dir = _get_trained_model_dir(tmp_path)
        result = push_predict.load_model(model_dir)
        assert len(result) == 4
        model, encoder_product, encoder_vendor, metadata = result
        assert model is not None
        assert encoder_product is not None
        assert encoder_vendor is not None
        assert isinstance(metadata, dict)
        assert "feature_cols" in metadata

    def test_load_model_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            push_predict.load_model(tmp_path / "nonexistent")


class TestPredict:
    def test_predict_returns_dict(self, tmp_path):
        model_dir = _get_trained_model_dir(tmp_path)
        predictions = push_predict.predict(SAMPLE_USERS, model_dir)
        assert isinstance(predictions, dict)
        assert len(predictions) == len(SAMPLE_USERS)
        for uid, pred in predictions.items():
            assert pred in (0, 1)

    def test_predict_raises_on_missing_columns(self, tmp_path):
        model_dir = _get_trained_model_dir(tmp_path)
        users = {"user_1": {"product_type": "x", "vendor": "y"}}
        with pytest.raises(ValueError, match="Missing columns"):
            push_predict.predict(users, model_dir)


class TestHandlerPredict:
    def test_handler_predict_returns_200_with_predictions(self, tmp_path):
        model_dir = _get_trained_model_dir(tmp_path)
        event = {
            "users": json.dumps(SAMPLE_USERS),
            "model_path": str(model_dir),
        }
        out = push_predict.handler_predict(event)
        assert out["statusCode"] == "200"
        body = json.loads(out["body"])
        assert "prediction" in body
        assert "user_1" in body["prediction"]
        assert body["prediction"]["user_1"] in (0, 1)

    def test_handler_predict_accepts_dict_users(self, tmp_path):
        model_dir = _get_trained_model_dir(tmp_path)
        event = {
            "users": SAMPLE_USERS,
            "model_path": str(model_dir),
        }
        out = push_predict.handler_predict(event)
        assert out["statusCode"] == "200"

    def test_handler_predict_returns_400_when_users_missing(self):
        out = push_predict.handler_predict({})
        assert out["statusCode"] == "400"
        body = json.loads(out["body"])
        assert "error" in body

    def test_handler_predict_returns_500_on_invalid_model_path(self):
        event = {
            "users": json.dumps(SAMPLE_USERS),
            "model_path": "/nonexistent/path",
        }
        out = push_predict.handler_predict(event)
        assert out["statusCode"] == "500"
        body = json.loads(out["body"])
        assert "error" in body
