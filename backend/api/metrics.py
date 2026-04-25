import time
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response
from typing import Optional

router = APIRouter(tags=["metrics"])

# Define metrics
requests_total = Counter("costguard_requests_total", "Total HTTP requests", ["method", "endpoint"])
cost_anomalies_total = Counter("costguard_anomalies_total", "Total cost anomalies detected", ["decision"])
active_pipelines = Gauge("costguard_active_pipelines", "Number of pipelines currently executing")
current_crs_score = Gauge("costguard_current_crs", "Current highest CRS score across pipelines")
request_duration = Histogram("costguard_request_duration_seconds", "HTTP request duration", ["method", "endpoint"])

@router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Helper functions to update metrics
def increment_request(method: str, endpoint: str):
    requests_total.labels(method=method, endpoint=endpoint).inc()

def record_anomaly(decision: str):
    cost_anomalies_total.labels(decision=decision).inc()

def set_active_pipelines(count: int):
    active_pipelines.set(count)

def set_current_crs(crs: float):
    current_crs_score.set(crs)

# Middleware to track request duration (to be added in main.py)
async def metrics_middleware(request, call_next):
    method = request.method
    path = request.url.path
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    request_duration.labels(method=method, endpoint=path).observe(duration)
    increment_request(method, path)
    return response