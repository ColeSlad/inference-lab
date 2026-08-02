import json

from inference_lab.benchmark.stats import percentile, summarize_repetitions, summarize_results


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
    assert summary["failure_rate"] == 0.333333
    assert summary["request_throughput_rps"] == 1.0
    assert summary["output_throughput_tokens_per_s"] == 20.0


def test_serialized_trial_retains_throughput_denominator() -> None:
    rows = [
        {"status": "ok", "ttft_ms": 10, "total_latency_ms": 100, "output_tokens": 64}
    ]
    summary = summarize_results(rows, concurrency=1, wall_time_s=4.023987654321)
    serialized = json.loads(json.dumps(summary))
    recomputed = summarize_results(
        rows,
        concurrency=1,
        wall_time_s=serialized["wall_time_s"],
    )

    assert serialized["wall_time_s"] == 4.023987654321
    assert recomputed["request_throughput_rps"] == serialized["request_throughput_rps"]
    assert (
        recomputed["output_throughput_tokens_per_s"]
        == serialized["output_throughput_tokens_per_s"]
    )


def test_summary_does_not_invent_missing_token_counts() -> None:
    rows = [
        {"status": "ok", "ttft_ms": 10, "total_latency_ms": 100, "output_tokens": 20},
        {"status": "ok", "ttft_ms": 20, "total_latency_ms": 200, "output_tokens": None},
    ]

    summary = summarize_results(rows, concurrency=2, wall_time_s=2.0)

    assert summary["missing_output_token_counts"] == 1
    assert summary["output_throughput_tokens_per_s"] is None
    assert summary["mean_output_tokens"] is None


def test_repetition_summary_uses_medians() -> None:
    summaries = [
        summarize_results(
            [{"status": "ok", "ttft_ms": 10, "total_latency_ms": 100, "output_tokens": 10}],
            concurrency=1,
            wall_time_s=1.0,
        ),
        summarize_results(
            [{"status": "ok", "ttft_ms": 30, "total_latency_ms": 300, "output_tokens": 10}],
            concurrency=1,
            wall_time_s=2.0,
        ),
        summarize_results(
            [{"status": "ok", "ttft_ms": 20, "total_latency_ms": 200, "output_tokens": 10}],
            concurrency=1,
            wall_time_s=3.0,
        ),
    ]

    aggregate = summarize_repetitions(summaries)[0]

    assert aggregate["aggregation"] == "median_across_repetitions"
    assert aggregate["repetitions"] == 3
    assert aggregate["wall_time_s"] == 2.0
    assert aggregate["ttft_ms_p50"] == 20.0
