import argparse
import asyncio
import json
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import httpx

from inference_lab.benchmark.stats import summarize_results


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing non-empty 'prompt' on {path}:{line_number}")
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


async def run_one(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    if seed is not None:
        payload["seed"] = seed

    started = time.perf_counter()
    first_chunk_at: float | None = None
    output_text = ""
    final_event: dict[str, Any] = {}

    try:
        endpoint = f"{url.rstrip('/')}/v1/generate/stream"
        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data:
                    continue
                event = json.loads(data)
                if event.get("type") == "chunk":
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    output_text += event.get("text", "")
                elif event.get("type") == "done":
                    final_event = event
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "Unknown streaming error"))

        finished = time.perf_counter()
        return {
            "status": "ok",
            "prompt_chars": len(prompt),
            "output_chars": len(output_text),
            "prompt_tokens": final_event.get("prompt_tokens"),
            "output_tokens": final_event.get("output_tokens"),
            "ttft_ms": (
                (first_chunk_at - started) * 1000 if first_chunk_at is not None else None
            ),
            "total_latency_ms": (finished - started) * 1000,
            "backend": final_event.get("backend"),
            "model": final_event.get("model"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "prompt_chars": len(prompt),
            "error": str(exc),
            "total_latency_ms": (time.perf_counter() - started) * 1000,
        }


async def run_concurrency_level(
    *,
    url: str,
    prompts: list[str],
    request_count: int,
    concurrency: int,
    max_new_tokens: int,
    temperature: float,
    seed: int | None,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(concurrency)
    selected_prompts = list(islice(cycle(prompts), request_count))

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        async def guarded(prompt: str) -> dict[str, Any]:
            async with semaphore:
                return await run_one(
                    client,
                    url,
                    prompt,
                    max_new_tokens,
                    temperature,
                    seed,
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(prompt) for prompt in selected_prompts))
        wall_time_s = time.perf_counter() - started
    return results, wall_time_s


async def async_main(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.dataset)
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, object]] = []

    for concurrency in args.concurrency:
        results, wall_time_s = await run_concurrency_level(
            url=args.url,
            prompts=prompts,
            request_count=args.requests,
            concurrency=concurrency,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            timeout_s=args.timeout,
        )
        for index, row in enumerate(results):
            row.update({"concurrency": concurrency, "request_index": index})
            all_rows.append(row)

        summary = summarize_results(results, concurrency=concurrency, wall_time_s=wall_time_s)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    with output_path.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"runs": summaries}, indent=2) + "\n")
    print(f"Wrote raw results to {output_path}")
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Inference Lab streaming API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=Path("data/prompts.jsonl"))
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    args.concurrency = [int(value) for value in args.concurrency.split(",") if value.strip()]
    if not args.concurrency or any(value < 1 for value in args.concurrency):
        parser.error("--concurrency must contain positive integers")
    if args.requests < 1:
        parser.error("--requests must be positive")
    return args


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
