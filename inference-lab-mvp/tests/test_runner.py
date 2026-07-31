import argparse
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from inference_lab.benchmark import runner


@pytest.mark.asyncio
async def test_run_one_records_complete_stream() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        content = "\n\n".join(
            [
                'data: {"type":"chunk","text":"hello"}',
                (
                    'data: {"type":"done","backend":"mock","model":"test",'
                    '"prompt_tokens":2,"output_tokens":1,"ttft_ms":1,"total_latency_ms":2}'
                ),
            ]
        )
        return httpx.Response(200, text=f"{content}\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await runner.run_one(client, "http://test", "a prompt", 4, 0.0, 0.9, 42)

    assert result["status"] == "ok"
    assert result["output_chars"] == 5
    assert result["server_ttft_ms"] == 1
    assert seen_payload["top_p"] == 0.9


@pytest.mark.asyncio
async def test_run_one_rejects_stream_without_done_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='data: {"type":"chunk","text":"partial"}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await runner.run_one(client, "http://test", "prompt", 4, 0.0, 1.0, 42)

    assert result["status"] == "error"
    assert result["error"] == "Stream ended without a terminal done event"


@pytest.mark.asyncio
async def test_async_main_writes_reproducible_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('{"prompt":"first"}\n{"prompt":"second"}\n', encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text('{"hardware":{"gpu":"test-gpu"}}\n', encoding="utf-8")
    output = tmp_path / "raw.jsonl"
    calls: list[tuple[int, int]] = []

    async def fake_runtime_info(url: str, timeout_s: float) -> dict[str, object]:
        return {"backend": "mock", "model": "mock-test"}

    async def fake_concurrency_level(**kwargs: object) -> tuple[list[dict[str, object]], float]:
        request_count = int(kwargs["request_count"])
        concurrency = int(kwargs["concurrency"])
        calls.append((request_count, concurrency))
        return (
            [
                {
                    "status": "ok",
                    "ttft_ms": 10.0,
                    "total_latency_ms": 20.0,
                    "output_tokens": 4,
                }
                for _ in range(request_count)
            ],
            1.0,
        )

    monkeypatch.setattr(runner, "fetch_runtime_info", fake_runtime_info)
    monkeypatch.setattr(runner, "run_concurrency_level", fake_concurrency_level)
    monkeypatch.setattr(runner, "git_metadata", lambda cwd: {"commit": "abc123", "dirty": False})
    monkeypatch.setattr(runner, "package_versions", lambda: {"inference-lab": "test"})
    args = argparse.Namespace(
        url="http://test",
        dataset=dataset,
        concurrency=[1, 2],
        requests=2,
        repetitions=2,
        warmup_requests=1,
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        timeout=10.0,
        metadata=metadata,
        output=output,
        command="inference-lab-bench --test",
        process_argv=["python", "-m", "inference_lab.benchmark.runner", "--test"],
    )

    await runner.async_main(args)

    raw_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert calls == [(1, 1), (2, 1), (1, 2), (2, 2)] * 2
    assert len(raw_rows) == 8
    assert {row["repetition"] for row in raw_rows} == {1, 2}
    assert raw_rows[0]["prompt_sha256"] == hashlib.sha256(b"first").hexdigest()
    assert summary["manifest"]["dataset"]["sha256"] == runner.sha256_file(dataset)
    assert summary["manifest"]["git"] == {"commit": "abc123", "dirty": False}
    assert summary["manifest"]["process_argv"][1:3] == [
        "-m",
        "inference_lab.benchmark.runner",
    ]
    assert summary["manifest"]["runtime"]["gateway_url"] == "http://test"
    assert summary["manifest"]["user_metadata"]["hardware"]["gpu"] == "test-gpu"
    assert len(summary["trials"]) == 4
    assert len(summary["runs"]) == 2
    assert summary["runs"][0]["repetitions"] == 2
