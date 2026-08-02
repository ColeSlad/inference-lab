import json
from pathlib import Path

import pytest

from inference_lab.benchmark.plotting import (
    PlotSeries,
    load_series,
    needs_log_scale,
    repetition_error_bars,
)


def write_summary(path: Path, concurrency: int) -> None:
    path.write_text(
        json.dumps({"runs": [{"concurrency": concurrency}]}),
        encoding="utf-8",
    )


def test_load_series_uses_explicit_labels(tmp_path: Path) -> None:
    first = tmp_path / "first.summary.json"
    second = tmp_path / "second.summary.json"
    write_summary(first, 1)
    write_summary(second, 2)

    series = load_series([first, second], ["Transformers", "vLLM"])

    assert series == [
        PlotSeries("Transformers", [{"concurrency": 1}], []),
        PlotSeries("vLLM", [{"concurrency": 2}], []),
    ]


def test_load_series_rejects_mismatched_labels(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    write_summary(summary, 1)

    with pytest.raises(ValueError, match="exactly one"):
        load_series([summary], ["first", "extra"])


def test_load_series_rejects_empty_runs(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text('{"runs": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="no aggregate runs"):
        load_series([summary], None)


def test_log_scale_is_used_for_large_cross_series_range() -> None:
    series = [
        PlotSeries("baseline", [{"throughput": 25.0}], []),
        PlotSeries("optimized", [{"throughput": 2_000.0}], []),
    ]

    assert needs_log_scale(series, "throughput") is True
    assert needs_log_scale(series, "throughput", ratio_threshold=100) is False


def test_repetition_error_bars_use_min_and_max() -> None:
    series = PlotSeries(
        "backend",
        [{"concurrency": 1, "throughput": 20.0}],
        [
            {"concurrency": 1, "throughput": 18.0},
            {"concurrency": 1, "throughput": 20.0},
            {"concurrency": 1, "throughput": 23.0},
        ],
    )

    assert repetition_error_bars(series, "throughput") == ([2.0], [3.0])
