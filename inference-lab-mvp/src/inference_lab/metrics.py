from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "inference_lab_requests_total",
    "Total inference requests.",
    labelnames=("backend", "mode", "status"),
)
IN_FLIGHT = Gauge(
    "inference_lab_in_flight_requests",
    "Requests currently being processed.",
    labelnames=("backend", "mode"),
)
REQUEST_LATENCY = Histogram(
    "inference_lab_request_latency_seconds",
    "End-to-end request latency.",
    labelnames=("backend", "mode"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
TTFT = Histogram(
    "inference_lab_ttft_seconds",
    "Time to first streamed text chunk.",
    labelnames=("backend",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
GENERATED_TOKENS = Counter(
    "inference_lab_generated_tokens_total",
    "Generated output tokens.",
    labelnames=("backend",),
)
