import math
from collections import defaultdict
from collections.abc import Iterable

SUMMARY_METRICS = (
    "wall_time_s",
    "request_throughput_rps",
    "output_throughput_tokens_per_s",
    "mean_output_tokens",
    "failure_rate",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "latency_ms_p50",
    "latency_ms_p95",
)


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    samples = sorted(values)
    if not samples:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if len(samples) == 1:
        return float(samples[0])

    position = (len(samples) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(samples[lower])
    weight = position - lower
    return float(samples[lower] * (1 - weight) + samples[upper] * weight)


def summarize_results(
    results: list[dict[str, object]], *, concurrency: int, wall_time_s: float
) -> dict[str, object]:
    successes = [row for row in results if row.get("status") == "ok"]
    ttft_values = [float(row["ttft_ms"]) for row in successes if row.get("ttft_ms") is not None]
    latency_values = [float(row["total_latency_ms"]) for row in successes]
    rows_with_output_tokens = [row for row in successes if row.get("output_tokens") is not None]
    output_tokens = sum(int(row["output_tokens"]) for row in rows_with_output_tokens)
    token_counts_complete = len(rows_with_output_tokens) == len(successes)

    safe_wall_time = max(wall_time_s, 1e-9)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(results) - len(successes),
        "failure_rate": round((len(results) - len(successes)) / len(results), 6)
        if results
        else 0.0,
        # Preserve the timer value used for throughput so a serialized trial can be
        # recomputed exactly. Aggregate summaries still round medians for display.
        "wall_time_s": wall_time_s,
        "request_throughput_rps": round(len(successes) / safe_wall_time, 4),
        "output_throughput_tokens_per_s": (
            round(output_tokens / safe_wall_time, 4) if token_counts_complete else None
        ),
        "mean_output_tokens": (
            round(output_tokens / len(successes), 2)
            if successes and token_counts_complete
            else None
        ),
        "missing_output_token_counts": len(successes) - len(rows_with_output_tokens),
        "ttft_ms_p50": _rounded(percentile(ttft_values, 50)),
        "ttft_ms_p95": _rounded(percentile(ttft_values, 95)),
        "latency_ms_p50": _rounded(percentile(latency_values, 50)),
        "latency_ms_p95": _rounded(percentile(latency_values, 95)),
    }


def summarize_repetitions(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-repetition summaries using the experiment plan's median rule."""
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for summary in summaries:
        grouped[int(summary["concurrency"])].append(summary)

    aggregated: list[dict[str, object]] = []
    for concurrency in sorted(grouped):
        repetitions = grouped[concurrency]
        request_counts = {int(summary["requests"]) for summary in repetitions}
        if len(request_counts) != 1:
            raise ValueError("All repetitions must use the same request count")

        row: dict[str, object] = {
            "concurrency": concurrency,
            "aggregation": "median_across_repetitions",
            "repetitions": len(repetitions),
            "requests_per_repetition": request_counts.pop(),
        }
        for field in SUMMARY_METRICS:
            values = [
                float(summary[field])
                for summary in repetitions
                if summary.get(field) is not None
            ]
            row[field] = (
                _rounded(percentile(values, 50))
                if len(values) == len(repetitions)
                else None
            )
        aggregated.append(row)
    return aggregated


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None
