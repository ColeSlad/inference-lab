import argparse
import asyncio
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import httpx

from inference_lab.benchmark.stats import summarize_repetitions, summarize_results

ARTIFACT_SCHEMA_VERSION = 1
RECORDED_PACKAGES = ("inference-lab", "httpx", "pydantic")


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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


def load_user_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in metadata file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(cwd: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def package_versions() -> dict[str, str | None]:
    packages: dict[str, str | None] = {}
    for package in RECORDED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


async def fetch_runtime_info(url: str, timeout_s: float) -> dict[str, object]:
    endpoint = f"{url.rstrip('/')}/health"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        response = await client.get(endpoint)
        response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or body.get("ok") is not True:
        raise RuntimeError(f"Backend health check did not report ok: {body!r}")
    return {key: value for key, value in body.items() if key != "ok"}


async def run_one(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> dict[str, Any]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed

    request_started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    first_chunk_at: float | None = None
    output_text = ""
    final_event: dict[str, Any] | None = None

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
                    text = event.get("text", "")
                    if text:
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                        output_text += text
                elif event.get("type") == "done":
                    final_event = event
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "Unknown streaming error"))

        if final_event is None:
            raise RuntimeError("Stream ended without a terminal done event")

        finished = time.perf_counter()
        return {
            "status": "ok",
            "request_started_at_utc": request_started_at,
            "prompt_chars": len(prompt),
            "output_chars": len(output_text),
            "prompt_tokens": final_event.get("prompt_tokens"),
            "output_tokens": final_event.get("output_tokens"),
            "ttft_ms": (
                (first_chunk_at - started) * 1000 if first_chunk_at is not None else None
            ),
            "total_latency_ms": (finished - started) * 1000,
            "server_ttft_ms": final_event.get("ttft_ms"),
            "server_total_latency_ms": final_event.get("total_latency_ms"),
            "backend": final_event.get("backend"),
            "model": final_event.get("model"),
        }
    except Exception as exc:
        return {
            "status": "error",
            "request_started_at_utc": request_started_at,
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
    top_p: float,
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
                    top_p,
                    seed,
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(prompt) for prompt in selected_prompts))
        wall_time_s = time.perf_counter() - started
    return results, wall_time_s


def build_manifest(
    args: argparse.Namespace,
    *,
    experiment_id: str,
    prompts: list[str],
    runtime: dict[str, object],
    started_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "started_at_utc": started_at.isoformat(),
        "command": args.command,
        "process_argv": args.process_argv,
        "git": git_metadata(Path.cwd()),
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256_file(args.dataset),
            "prompt_count": len(prompts),
        },
        "runtime": {"gateway_url": args.url.rstrip("/"), **runtime},
        "request": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "benchmark": {
            "concurrency": args.concurrency,
            "requests_per_repetition": args.requests,
            "repetitions": args.repetitions,
            "warmup_requests_before_each_repetition": args.warmup_requests,
            "timeout_s": args.timeout,
        },
        "client_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": package_versions(),
        },
        "user_metadata": load_user_metadata(args.metadata),
    }


async def async_main(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.dataset)
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    experiment_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    runtime = await fetch_runtime_info(args.url, args.timeout)
    manifest = build_manifest(
        args,
        experiment_id=experiment_id,
        prompts=prompts,
        runtime=runtime,
        started_at=started_at,
    )
    trial_summaries: list[dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as raw_output:
        for repetition in range(1, args.repetitions + 1):
            for concurrency in args.concurrency:
                if args.warmup_requests:
                    warmups, _ = await run_concurrency_level(
                        url=args.url,
                        prompts=prompts,
                        request_count=args.warmup_requests,
                        concurrency=concurrency,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                        timeout_s=args.timeout,
                    )
                    warmup_failures = [row for row in warmups if row.get("status") != "ok"]
                    if warmup_failures:
                        raise RuntimeError(
                            f"{len(warmup_failures)} warm-up request(s) failed at "
                            f"concurrency {concurrency}, repetition {repetition}"
                        )

                results, wall_time_s = await run_concurrency_level(
                    url=args.url,
                    prompts=prompts,
                    request_count=args.requests,
                    concurrency=concurrency,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    timeout_s=args.timeout,
                )
                for request_index, row in enumerate(results):
                    prompt_index = request_index % len(prompts)
                    row.update(
                        {
                            "experiment_id": experiment_id,
                            "repetition": repetition,
                            "concurrency": concurrency,
                            "request_index": request_index,
                            "prompt_index": prompt_index,
                            "prompt_sha256": hashlib.sha256(
                                prompts[prompt_index].encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                    raw_output.write(json.dumps(row, sort_keys=True) + "\n")
                raw_output.flush()

                summary = summarize_results(
                    results,
                    concurrency=concurrency,
                    wall_time_s=wall_time_s,
                )
                summary["repetition"] = repetition
                trial_summaries.append(summary)
                print(json.dumps(summary, indent=2))

    summary_path = output_path.with_suffix(".summary.json")
    summary_document = {
        "manifest": {
            **manifest,
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
        "runs": summarize_repetitions(trial_summaries),
        "trials": trial_summaries,
    }
    summary_path.write_text(
        json.dumps(summary_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote raw results to {output_path}")
    print(f"Wrote summary and manifest to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Inference Lab streaming API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=Path("data/prompts.jsonl"))
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="JSON object with server hardware, image digest, model revision, and runtime flags.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/latest.jsonl"))
    args = parser.parse_args()
    args.command = shlex.join(["inference-lab-bench", *sys.argv[1:]])
    args.process_argv = list(sys.orig_argv)
    args.concurrency = [int(value) for value in args.concurrency.split(",") if value.strip()]
    if not args.concurrency or any(value < 1 for value in args.concurrency):
        parser.error("--concurrency must contain positive integers")
    if len(args.concurrency) != len(set(args.concurrency)):
        parser.error("--concurrency must not contain duplicates")
    if args.requests < 1:
        parser.error("--requests must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
