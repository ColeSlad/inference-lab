from inference_lab.benchmark.stats import percentile, summarize_results


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_summary_calculates_throughput() -> None:
    rows = [
        {"status": "ok", "ttft_ms": 10, "total_latency_ms": 100, "output_tokens": 20},
        {"status": "ok", "ttft_ms": 20, "total_latency_ms": 200, "output_tokens": 20},
        {"status": "error", "total_latency_ms": 50},
    ]
    summary = summarize_results(rows, concurrency=2, wall_time_s=2.0)
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["request_throughput_rps"] == 1.0
    assert summary["output_throughput_tokens_per_s"] == 20.0
