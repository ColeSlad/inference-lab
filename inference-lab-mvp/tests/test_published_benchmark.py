import hashlib
import json
from pathlib import Path

from inference_lab.benchmark.runner import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "benchmarks" / "2026-08-01-qwen3-8b-a100"
EXPECTED_COMMIT = "4983d06d8076b201dd74b2a85df4009ddd790eff"
EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def test_published_artifact_checksums() -> None:
    for line in (BENCHMARK / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split(maxsplit=1)
        artifact = BENCHMARK / relative_path

        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected


def test_published_comparison_has_matching_controls_and_complete_rows() -> None:
    dataset = PROJECT_ROOT / "data" / "short_chat.jsonl"
    token_report = json.loads(
        (PROJECT_ROOT / "data" / "short_chat.qwen3-8b.tokens.json").read_text(
            encoding="utf-8"
        )
    )
    expected_prompt_tokens = token_report["observed_prompt_tokens"]["per_prompt"]

    for backend in ("transformers", "vllm"):
        summary = json.loads(
            (BENCHMARK / f"{backend}.summary.json").read_text(encoding="utf-8")
        )
        rows = [
            json.loads(line)
            for line in (BENCHMARK / f"{backend}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        manifest = summary["manifest"]

        assert manifest["git"] == {"commit": EXPECTED_COMMIT, "dirty": False}
        assert manifest["dataset"]["sha256"] == sha256_file(dataset)
        assert manifest["runtime"]["model"] == "Qwen/Qwen3-8B"
        assert manifest["runtime"]["model_revision"] == EXPECTED_MODEL_REVISION
        assert manifest["runtime"]["model_dtype"] == "bfloat16"
        assert manifest["request"] == {
            "max_new_tokens": 64,
            "seed": 42,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        assert manifest["benchmark"]["concurrency"] == [1, 2, 4, 8, 16, 32]
        assert manifest["benchmark"]["requests_per_repetition"] == 120
        assert manifest["benchmark"]["repetitions"] == 3
        assert manifest["benchmark"]["warmup_requests_before_each_repetition"] == 10
        assert len(rows) == 2_160
        assert len(summary["trials"]) == 18
        assert len(summary["runs"]) == 6
        assert {row["status"] for row in rows} == {"ok"}
        assert {row["output_tokens"] for row in rows} == {64}
        for prompt_index, expected in enumerate(expected_prompt_tokens):
            assert {
                row["prompt_tokens"]
                for row in rows
                if row["prompt_index"] == prompt_index
            } == {expected}
