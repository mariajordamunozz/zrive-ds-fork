import logging
import time

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .basket_model.basket_model import BasketModel
from .basket_model.feature_store import FeatureStore
from .exceptions import PredictionException, UserNotFoundException
from .metrics import MetricsLogger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PredictRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="User identifier")


class PredictResponse(BaseModel):
    user_id: str
    predicted_basket_value: float
    inference_ms: float
    feature_lookup_ms: float


class ReadyResponse(BaseModel):
    status: str
    model: bool
    feature_store: bool


def create_app(
    feature_store: FeatureStore | None = None,
    model: BasketModel | None = None,
    metrics: MetricsLogger | None = None,
) -> FastAPI:
    """Build the FastAPI app. Dependencies are injectable so tests can pass fakes."""
    app = FastAPI(title="Basket price prediction API", version="1.0.0")

    app.state.feature_store = feature_store
    app.state.model = model
    app.state.metrics = metrics or MetricsLogger()

    @app.middleware("http")
    async def record_latency(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except HTTPException:
            raise
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            app.state.metrics.log_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                latency_ms=latency_ms,
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        app.state.metrics.log_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return response

    @app.get("/status")
    def status() -> dict[str, str]:
        """Liveness probe: confirms the process is up and serving requests."""
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> JSONResponse:
        """Readiness probe: confirms model and feature store are loaded."""
        model_ready = app.state.model is not None
        store_ready = app.state.feature_store is not None
        if model_ready and store_ready:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ready",
                    "model": True,
                    "feature_store": True,
                },
            )
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model": model_ready,
                "feature_store": store_ready,
            },
        )

    @app.get("/metrics")
    def metrics_endpoint() -> dict[str, object]:
        """Return cumulative in-memory counters for inspection."""
        return app.state.metrics.get_counters()

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if app.state.feature_store is None or app.state.model is None:
            app.state.metrics.log_error(
                endpoint="/predict",
                error_type="ServiceUnavailable",
                message="Model or feature store not initialised",
            )
            raise HTTPException(status_code=503, detail="Service not ready")

        user_id = payload.user_id

        lookup_start = time.perf_counter()
        try:
            features = app.state.feature_store.get_features(user_id)
        except Exception as exc:
            if "not in store" in str(exc):
                app.state.metrics.log_error(
                    endpoint="/predict",
                    error_type="UserNotFound",
                    message=str(exc),
                )
                raise HTTPException(status_code=404, detail="User not found") from exc
            else:
                raise
        feature_lookup_ms = (time.perf_counter() - lookup_start) * 1000

        try:
            if isinstance(features, pd.DataFrame):
                features = features.iloc[-1]
            features_array = np.asarray(features, dtype=float).reshape(1, -1)
            inference_start = time.perf_counter()
            prediction = float(app.state.model.predict(features_array)[0])
            inference_ms = (time.perf_counter() - inference_start) * 1000
        except PredictionException as exc:
            app.state.metrics.log_error(
                endpoint="/predict",
                error_type="PredictionFailed",
                message=str(exc),
            )
            raise HTTPException(status_code=500, detail="Prediction failed") from exc
        except Exception as exc:
            app.state.metrics.log_error(
                endpoint="/predict",
                error_type="UnexpectedError",
                message=str(exc),
            )
            raise HTTPException(status_code=500, detail="Internal server error") from exc

        app.state.metrics.log_prediction(
            user_id=user_id,
            prediction=prediction,
            inference_ms=inference_ms,
            feature_lookup_ms=feature_lookup_ms,
            n_features=int(features_array.shape[1]),
        )
        return PredictResponse(
            user_id=user_id,
            predicted_basket_value=prediction,
            inference_ms=inference_ms,
            feature_lookup_ms=feature_lookup_ms,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error at %s", request.url.path)
        app.state.metrics.log_error(
            endpoint=request.url.path,
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


def _build_default_app() -> FastAPI:
    feature_store = FeatureStore()
    model = BasketModel()
    return create_app(feature_store=feature_store, model=model)


app = _build_default_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
