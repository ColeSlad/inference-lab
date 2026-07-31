import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an Inference Lab summary JSON file.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    runs = json.loads(args.summary.read_text())["runs"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    concurrency = [row["concurrency"] for row in runs]

    figures = [
        ("request_throughput_rps", "Request throughput", "Requests / second"),
        ("output_throughput_tokens_per_s", "Output-token throughput", "Tokens / second"),
        ("ttft_ms_p95", "P95 time to first token", "Milliseconds"),
        ("latency_ms_p95", "P95 end-to-end latency", "Milliseconds"),
    ]

    for field, title, ylabel in figures:
        values = [row[field] for row in runs]
        plt.figure(figsize=(7, 4.5))
        plt.plot(concurrency, values, marker="o")
        plt.xlabel("Concurrency")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        output = args.output_dir / f"{field}.png"
        plt.savefig(output, dpi=160)
        plt.close()
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
