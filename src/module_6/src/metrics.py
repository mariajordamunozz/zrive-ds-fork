import os
import threading
from collections import defaultdict
from datetime import datetime, timezone

METRICS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "metrics.txt")
)


class MetricsLogger:
    """
    A thread-safe utility for logging metrics to a text file. Each event is 
    appended as a discrete line, ensuring data integrity during concurrent write operations.
    """

    def __init__(self, file_path: str = METRICS_FILE) -> None:
        self.file_path = file_path
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "requests_total": 0,
            "predictions_total": 0,
            "errors_total": 0,
        }
        self._errors_by_type: dict[str, int] = defaultdict(int)
        self._requests_by_status: dict[str, int] = defaultdict(int)

    def log(self, event: str, **fields: object) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        parts = [f"timestamp={timestamp}", f"event={event}"]
        parts.extend(f"{key}={value}" for key, value in fields.items())
        line = " | ".join(parts) + "\n"
        with self._lock:
            with open(self.file_path, "a", encoding="utf-8") as file:
                file.write(line)

    def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._counters["requests_total"] += 1
            self._requests_by_status[str(status_code)] += 1
        self.log(
            "request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=f"{latency_ms:.2f}",
        )

    def log_prediction(
        self,
        user_id: str,
        prediction: float,
        inference_ms: float | None = None,
        feature_lookup_ms: float | None = None,
        n_features: int | None = None,
    ) -> None:
        with self._lock:
            self._counters["predictions_total"] += 1
        fields: dict[str, object] = {
            "user_id": user_id,
            "prediction": f"{prediction:.4f}",
        }
        if inference_ms is not None:
            fields["inference_ms"] = f"{inference_ms:.2f}"
        if feature_lookup_ms is not None:
            fields["feature_lookup_ms"] = f"{feature_lookup_ms:.2f}"
        if n_features is not None:
            fields["n_features"] = n_features
        self.log("prediction", **fields)

    def log_error(self, endpoint: str, error_type: str, message: str) -> None:
        with self._lock:
            self._counters["errors_total"] += 1
            self._errors_by_type[error_type] += 1
        safe_message = message.replace("\n", " ").replace("|", "/")
        self.log(
            "error",
            endpoint=endpoint,
            error_type=error_type,
            message=safe_message,
        )

    def get_counters(self) -> dict[str, object]:
        """Return a snapshot of the in-memory cumulative counters."""
        with self._lock:
            return {
                **self._counters,
                "errors_by_type": dict(self._errors_by_type),
                "requests_by_status": dict(self._requests_by_status),
            }
