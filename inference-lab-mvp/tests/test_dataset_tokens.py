import json
from pathlib import Path

from inference_lab.benchmark.dataset_tokens import build_token_report
from inference_lab.benchmark.runner import load_prompts, sha256_file


def test_token_report_records_counts_and_tokenizer_identity() -> None:
    prompts = ["one", "three", "seven"]
    report = build_token_report(
        prompts,
        lambda prompt: list(range(len(prompt))),
        dataset=Path("data/test.jsonl"),
        dataset_sha256="abc123",
        model="test/model",
        revision="revision123",
        minimum=3,
        maximum=5,
    )

    assert report["observed_prompt_tokens"] == {
        "minimum": 3,
        "maximum": 5,
        "mean": 4.333,
        "per_prompt": [3, 5, 5],
    }
    assert report["valid"] is True
    assert report["violations"] == []
    assert report["tokenizer"]["revision"] == "revision123"


def test_token_report_identifies_prompt_indices_outside_bounds() -> None:
    report = build_token_report(
        ["a", "too long"],
        lambda prompt: list(range(len(prompt))),
        dataset=Path("data/test.jsonl"),
        dataset_sha256="abc123",
        model="test/model",
        revision="revision123",
        minimum=2,
        maximum=5,
    )

    assert report["valid"] is False
    assert report["violations"] == [
        {"prompt_index": 0, "prompt_tokens": 1},
        {"prompt_index": 1, "prompt_tokens": 8},
    ]


def test_short_chat_dataset_is_nonempty_and_unique() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "data" / "short_chat.jsonl"
    report_path = project_root / "data" / "short_chat.qwen3-8b.tokens.json"
    prompts = load_prompts(dataset)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(prompts) == 10
    assert len(set(prompts)) == len(prompts)
    assert report["valid"] is True
    assert report["dataset"]["prompt_count"] == len(prompts)
    assert report["dataset"]["sha256"] == sha256_file(dataset)
    assert len(report["observed_prompt_tokens"]["per_prompt"]) == len(prompts)
    assert (
        report["observed_prompt_tokens"]["minimum"]
        >= report["expected_prompt_tokens"]["minimum"]
    )
    assert (
        report["observed_prompt_tokens"]["maximum"]
        <= report["expected_prompt_tokens"]["maximum"]
    )
