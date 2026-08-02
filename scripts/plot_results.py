import argparse
from pathlib import Path

from inference_lab.benchmark.plotting import (
    load_series,
    needs_log_scale,
    repetition_error_bars,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one or more Inference Lab summaries.")
    parser.add_argument("summary", type=Path, nargs="+")
    parser.add_argument(
        "--label",
        action="append",
        help="Series label; provide once per summary (defaults to each filename).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    try:
        series = load_series(args.summary, args.label)
    except ValueError as exc:
        parser.error(str(exc))

    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("request_throughput_rps", "Request throughput", "Requests / second"),
        ("output_throughput_tokens_per_s", "Output-token throughput", "Tokens / second"),
        ("ttft_ms_p95", "P95 time to first token", "Milliseconds"),
        ("latency_ms_p95", "P95 end-to-end latency", "Milliseconds"),
    ]
    concurrency_levels = sorted(
        {int(row["concurrency"]) for item in series for row in item.runs}
    )

    for field, title, ylabel in figures:
        plt.figure(figsize=(7, 4.5))
        for item in series:
            concurrency = [row["concurrency"] for row in item.runs]
            values = [row[field] for row in item.runs]
            errors = repetition_error_bars(item, field)
            if errors is not None:
                plt.errorbar(
                    concurrency,
                    values,
                    yerr=errors,
                    marker="o",
                    capsize=3,
                    label=item.label,
                )
            else:
                plt.plot(concurrency, values, marker="o", label=item.label)
        plt.xlabel("Concurrency")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xscale("log", base=2)
        plt.xticks(concurrency_levels, [str(value) for value in concurrency_levels])
        if needs_log_scale(series, field):
            plt.yscale("log")
            plt.ylabel(f"{ylabel} (log scale)")
        plt.grid(True, alpha=0.25)
        if len(series) > 1:
            plt.legend()
        plt.tight_layout()
        output = args.output_dir / f"{field}.png"
        plt.savefig(output, dpi=160)
        plt.close()
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
