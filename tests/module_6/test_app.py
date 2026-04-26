import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.module_6.src.app import create_app
from src.module_6.src.exceptions import PredictionException, UserNotFoundException
from src.module_6.src.metrics import MetricsLogger


class FakeFeatureStore:
    def __init__(self, known_users: dict[str, list[float]]):
        self._known_users = known_users

    def get_features(self, user_id: str) -> pd.Series:
        if user_id not in self._known_users:
            raise UserNotFoundException(f"User {user_id} not in store")
        return pd.Series(self._known_users[user_id])


class FakeModel:
    def __init__(self, return_value: float = 42.0, raise_error: bool = False):
        self.return_value = return_value
        self.raise_error = raise_error

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.raise_error:
            raise PredictionException("boom")
        return np.array([self.return_value] * features.shape[0])


@pytest.fixture
def metrics_file(tmp_path):
    return tmp_path / "metrics.txt"


@pytest.fixture
def client(metrics_file):
    feature_store = FakeFeatureStore({"user-1": [10.0, 5.0, 2.0, 3.0]})
    model = FakeModel(return_value=99.5)
    metrics = MetricsLogger(file_path=str(metrics_file))
    app = create_app(feature_store=feature_store, model=model, metrics=metrics)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_exceptions(metrics_file):
    feature_store = FakeFeatureStore({"user-1": [10.0, 5.0, 2.0, 3.0]})
    model = FakeModel(return_value=99.5)
    metrics = MetricsLogger(file_path=str(metrics_file))
    app = create_app(feature_store=feature_store, model=model, metrics=metrics)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

def test_status_returns_200(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_dependencies_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ready", "model": True, "feature_store": True}


def test_ready_returns_503_when_dependencies_missing(metrics_file):
    metrics = MetricsLogger(file_path=str(metrics_file))
    app = create_app(feature_store=None, model=None, metrics=metrics)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["model"] is False
    assert body["feature_store"] is False


def test_predict_returns_prediction(client):
    response = client.post("/predict", json={"user_id": "user-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user-1"
    assert body["predicted_basket_value"] == pytest.approx(99.5)
    assert body["inference_ms"] >= 0
    assert body["feature_lookup_ms"] >= 0


def test_predict_unknown_user_returns_404(client_no_exceptions):
    response = client_no_exceptions.post("/predict", json={"user_id": "missing"})
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

    

def test_predict_missing_body_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_empty_user_id_returns_422(client):
    response = client.post("/predict", json={"user_id": ""})
    assert response.status_code == 422


def test_prediction_failure_returns_500(metrics_file):
    feature_store = FakeFeatureStore({"user-1": [1.0, 2.0, 3.0, 4.0]})
    model = FakeModel(raise_error=True)
    metrics = MetricsLogger(file_path=str(metrics_file))
    app = create_app(feature_store=feature_store, model=model, metrics=metrics)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"user_id": "user-1"})
    assert response.status_code == 500


def test_metrics_are_written_to_file(client_no_exceptions, metrics_file):
    client_no_exceptions.get("/status")
    client_no_exceptions.post("/predict", json={"user_id": "user-1"})
    client_no_exceptions.post("/predict", json={"user_id": "missing"})

    content = metrics_file.read_text()
    assert "event=request" in content
    assert "endpoint=/status" in content
    assert "endpoint=/predict" in content
    assert "event=prediction" in content
    assert "inference_ms=" in content
    assert "feature_lookup_ms=" in content
    assert "event=error" in content
    assert "error_type=UserNotFound" in content
    assert "method=GET" in content
    assert "method=POST" in content


def test_metrics_endpoint_returns_counters(client_no_exceptions):
    client_no_exceptions.get("/status")
    client_no_exceptions.post("/predict", json={"user_id": "user-1"})
    client_no_exceptions.post("/predict", json={"user_id": "missing"})

    response = client_no_exceptions.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["predictions_total"] == 1
    assert body["errors_total"] >= 1
    assert body["errors_by_type"].get("UserNotFound") == 1
    # /status, /predict (200), /predict (404) and the /metrics call itself
    assert body["requests_total"] >= 3


def test_service_unavailable_when_model_missing(metrics_file):
    metrics = MetricsLogger(file_path=str(metrics_file))
    app = create_app(feature_store=None, model=None, metrics=metrics)
    with TestClient(app) as client:
        response = client.post("/predict", json={"user_id": "user-1"})
    assert response.status_code == 503
