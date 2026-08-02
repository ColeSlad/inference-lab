import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlotSeries:
    label: str
    runs: list[dict[str, object]]
    trials: list[dict[str, object]]


def load_series(
    summaries: list[Path], labels: list[str] | None
) -> list[PlotSeries]:
    if labels is not None and len(labels) != len(summaries):
        raise ValueError("Provide exactly one --label for each summary")

    resolved_labels = labels or [path.name.removesuffix(".summary.json") for path in summaries]
    series = []
    for label, path in zip(resolved_labels, summaries, strict=True):
        document = json.loads(path.read_text(encoding="utf-8"))
        runs = document.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"Summary has no aggregate runs: {path}")
        trials = document.get("trials", [])
        if not isinstance(trials, list):
            raise ValueError(f"Summary trials must be a list: {path}")
        series.append(PlotSeries(label=label, runs=runs, trials=trials))
    return series


def needs_log_scale(
    series: list[PlotSeries],
    field: str,
    *,
    ratio_threshold: float = 20,
) -> bool:
    values = [
        float(row[field])
        for item in series
        for row in item.runs
        if row.get(field) is not None
    ]
    return bool(values) and min(values) > 0 and max(values) / min(values) >= ratio_threshold


def repetition_error_bars(
    series: PlotSeries, field: str
) -> tuple[list[float], list[float]] | None:
    if not series.trials:
        return None

    lows: list[float] = []
    highs: list[float] = []
    for run in series.runs:
        concurrency = int(run["concurrency"])
        values = [
            float(trial[field])
            for trial in series.trials
            if int(trial["concurrency"]) == concurrency and trial.get(field) is not None
        ]
        if not values or run.get(field) is None:
            return None
        center = float(run[field])
        # Aggregate medians are display-rounded, so clamp sub-milliscale negative
        # deltas when a rounded center falls just outside the raw trial range.
        lows.append(max(0.0, center - min(values)))
        highs.append(max(0.0, max(values) - center))
    return lows, highs
