from src.module_6.src.metrics import MetricsLogger


def test_log_request_writes_expected_fields(tmp_path):
    metrics_file = tmp_path / "metrics.txt"
    logger = MetricsLogger(file_path=str(metrics_file))

    logger.log_request(
        endpoint="/status", method="GET", status_code=200, latency_ms=12.34
    )

    line = metrics_file.read_text().strip()
    assert "event=request" in line
    assert "endpoint=/status" in line
    assert "method=GET" in line
    assert "status_code=200" in line
    assert "latency_ms=12.34" in line


def test_log_prediction_writes_expected_fields(tmp_path):
    metrics_file = tmp_path / "metrics.txt"
    logger = MetricsLogger(file_path=str(metrics_file))

    logger.log_prediction(
        user_id="user-1",
        prediction=42.1234,
        inference_ms=1.23,
        feature_lookup_ms=4.56,
        n_features=4,
    )

    line = metrics_file.read_text().strip()
    assert "event=prediction" in line
    assert "user_id=user-1" in line
    assert "prediction=42.1234" in line
    assert "inference_ms=1.23" in line
    assert "feature_lookup_ms=4.56" in line
    assert "n_features=4" in line


def test_log_prediction_optional_fields_omitted(tmp_path):
    metrics_file = tmp_path / "metrics.txt"
    logger = MetricsLogger(file_path=str(metrics_file))

    logger.log_prediction(user_id="user-1", prediction=10.0)

    line = metrics_file.read_text().strip()
    assert "inference_ms" not in line
    assert "feature_lookup_ms" not in line
    assert "n_features" not in line


def test_log_error_sanitises_separator(tmp_path):
    metrics_file = tmp_path / "metrics.txt"
    logger = MetricsLogger(file_path=str(metrics_file))

    logger.log_error(
        endpoint="/predict",
        error_type="UserNotFound",
        message="broken|message\nwith newline",
    )

    content = metrics_file.read_text()
    assert content.count("\n") == 1
    assert "message=broken/message with newline" in content


def test_logger_appends_multiple_lines(tmp_path):
    metrics_file = tmp_path / "metrics.txt"
    logger = MetricsLogger(file_path=str(metrics_file))

    logger.log_request(endpoint="/a", method="GET", status_code=200, latency_ms=1.0)
    logger.log_request(endpoint="/b", method="GET", status_code=500, latency_ms=2.0)

    lines = metrics_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_counters_increment_on_each_event(tmp_path):
    metrics_file = tmp_path / "metrics.txt"
    logger = MetricsLogger(file_path=str(metrics_file))

    logger.log_request(endpoint="/a", method="GET", status_code=200, latency_ms=1.0)
    logger.log_request(endpoint="/b", method="POST", status_code=500, latency_ms=2.0)
    logger.log_prediction(user_id="u", prediction=1.0)
    logger.log_prediction(user_id="u", prediction=2.0)
    logger.log_error(endpoint="/x", error_type="UserNotFound", message="m")

    counters = logger.get_counters()
    assert counters["requests_total"] == 2
    assert counters["predictions_total"] == 2
    assert counters["errors_total"] == 1
    assert counters["errors_by_type"] == {"UserNotFound": 1}
    assert counters["requests_by_status"] == {"200": 1, "500": 1}
