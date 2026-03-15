"""Unit tests for push_train.py."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import push_train


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


class TestLoadData:
    def test_load_data_returns_dataframe(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        df = push_train.load_data(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_data_has_required_columns(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        df = push_train.load_data(path)
        for col in push_train.REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"


class TestValidateData:
    def test_validate_data_passes_clean_data(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        df = push_train.load_data(path)
        df_valid = push_train.validate_data(df)
        assert len(df_valid) == len(df)

    def test_validate_data_raises_on_missing_columns(self):
        df = pd.DataFrame({"order_id": [1], "user_id": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            push_train.validate_data(df)

    def test_validate_data_raises_on_nulls_by_default(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        df = push_train.load_data(path)
        df.loc[df.index[0], "outcome"] = None
        with pytest.raises(ValueError, match="Critical columns have nulls"):
            push_train.validate_data(df, on_null="raise")


class TestBuildModel:
    def test_build_logistic_regression(self):
        pipe = push_train._build_model({"model_type": "logistic_regression"})
        assert pipe is not None
        assert pipe.named_steps["clf"].__class__.__name__ == "LogisticRegression"

    def test_build_random_forest(self):
        pipe = push_train._build_model({"model_type": "random_forest"})
        assert pipe.named_steps["clf"].__class__.__name__ == "RandomForestClassifier"

    def test_build_gradient_boosting(self):
        pipe = push_train._build_model({"model_type": "gradient_boosting"})
        assert (
            pipe.named_steps["clf"].__class__.__name__ == "GradientBoostingClassifier"
        )

    def test_build_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            push_train._build_model({"model_type": "unknown"})


class TestPreprocess:
    def test_preprocess_returns_correct_shapes(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        df = push_train.load_data(path)
        df = push_train.validate_data(df)
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            artifacts,
        ) = push_train.preprocess(df)
        assert X_train.shape[0] == len(y_train)
        assert X_val.shape[0] == len(y_val)
        assert X_test.shape[0] == len(y_test)
        assert "encoder_product" in artifacts
        assert "encoder_vendor" in artifacts
        assert "feature_cols" in artifacts

    def test_preprocess_filters_orders_with_5_plus_items(self, tmp_path):
        """Sample has 20 orders: first 10 with ≥5 items (qualify), last 10 with 0 (excluded)."""
        path = _make_sample_feature_frame(tmp_path)
        df = push_train.load_data(path)
        df = push_train.validate_data(df)
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            artifacts,
        ) = push_train.preprocess(df)
        total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
        # Only 10 qualifying orders × 10 rows = 100 rows (orders 11-20 excluded)
        assert total_rows == 100


class TestTrainAndSave:
    def test_train_and_save_creates_model_files(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        result = push_train.train_and_save(
            data_path=path,
            output_dir=tmp_path,
            model_parametrisation={
                "model_type": "logistic_regression",
                "max_iter": 500,
            },
        )
        model_path = Path(result["model_path"])
        assert model_path.exists()
        assert (model_path / "model.pkl").exists()
        assert (model_path / "encoder_product.pkl").exists()
        assert (model_path / "encoder_vendor.pkl").exists()
        assert (model_path / "metadata.json").exists()
        assert "model_path" in result
        assert "model_name" in result
        assert result["model_name"].startswith("push_")

    def test_train_and_save_metadata_contains_metrics(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        result = push_train.train_and_save(
            data_path=path,
            output_dir=tmp_path,
            model_parametrisation={
                "model_type": "logistic_regression",
                "max_iter": 500,
            },
        )
        with open(Path(result["model_path"]) / "metadata.json") as f:
            metadata = json.load(f)
        assert "validation_precision" in metadata
        assert "validation_recall" in metadata
        assert "feature_cols" in metadata


class TestHandlerFit:
    def test_handler_fit_returns_200_with_model_path(self, tmp_path):
        path = _make_sample_feature_frame(tmp_path)
        event = {
            "data_path": str(path),
            "output_base": str(tmp_path),
            "model_parametrisation": {
                "model_type": "logistic_regression",
                "max_iter": 500,
            },
        }
        out = push_train.handler_fit(event)
        assert out["statusCode"] == "200"
        body = json.loads(out["body"])
        assert "model_path" in body
        assert "model_name" in body
