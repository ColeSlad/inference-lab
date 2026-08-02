import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

from inference_lab.benchmark.runner import load_prompts, sha256_file


def build_token_report(
    prompts: list[str],
    encode: Callable[[str], list[int]],
    *,
    dataset: Path,
    dataset_sha256: str,
    model: str,
    revision: str,
    minimum: int,
    maximum: int,
) -> dict[str, object]:
    if minimum < 0:
        raise ValueError("minimum token count must be non-negative")
    if maximum < minimum:
        raise ValueError("maximum token count must be at least the minimum")

    counts = [len(encode(prompt)) for prompt in prompts]
    violations = [
        {"prompt_index": index, "prompt_tokens": count}
        for index, count in enumerate(counts)
        if count < minimum or count > maximum
    ]
    return {
        "dataset": {
            "path": str(dataset),
            "sha256": dataset_sha256,
            "prompt_count": len(prompts),
        },
        "tokenizer": {
            "model": model,
            "revision": revision,
            "add_special_tokens": True,
        },
        "expected_prompt_tokens": {"minimum": minimum, "maximum": maximum},
        "observed_prompt_tokens": {
            "minimum": min(counts),
            "maximum": max(counts),
            "mean": round(statistics.mean(counts), 3),
            "per_prompt": counts,
        },
        "valid": not violations,
        "violations": violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset prompt lengths with an exact tokenizer revision."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--min-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            'Dataset token inspection requires: pip install -e ".[hf]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    prompts = load_prompts(args.dataset)
    report = build_token_report(
        prompts,
        lambda prompt: tokenizer.encode(prompt, add_special_tokens=True),
        dataset=args.dataset,
        dataset_sha256=sha256_file(args.dataset),
        model=args.model,
        revision=args.revision,
        minimum=args.min_tokens,
        maximum=args.max_tokens,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
