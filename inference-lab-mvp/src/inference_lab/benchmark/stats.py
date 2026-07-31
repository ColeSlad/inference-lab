import math
from collections.abc import Iterable


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
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in successes)

    safe_wall_time = max(wall_time_s, 1e-9)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(results) - len(successes),
        "wall_time_s": round(wall_time_s, 6),
        "request_throughput_rps": round(len(successes) / safe_wall_time, 4),
        "output_throughput_tokens_per_s": round(output_tokens / safe_wall_time, 4),
        "mean_output_tokens": round(output_tokens / len(successes), 2) if successes else 0,
        "ttft_ms_p50": _rounded(percentile(ttft_values, 50)),
        "ttft_ms_p95": _rounded(percentile(ttft_values, 95)),
        "latency_ms_p50": _rounded(percentile(latency_values, 50)),
        "latency_ms_p95": _rounded(percentile(latency_values, 95)),
    }


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None
