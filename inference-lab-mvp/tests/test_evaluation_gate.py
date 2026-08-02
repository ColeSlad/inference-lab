import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_lab.benchmark.gate import (
    EvaluationPolicy,
    build_evaluation_report,
    evaluate_files,
    evaluate_performance,
    load_policy,
    parse_args,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row(prompt: str, output: str, *, concurrency: int) -> dict[str, object]:
    return {
        "status": "ok",
        "prompt_sha256": digest(prompt),
        "concurrency": concurrency,
        "repetition": 1,
        "request_index": 0,
        "output_sha256": digest(output),
    }


def summary() -> dict[str, object]:
    return {
        "manifest": {
            "git": {"commit": "commit", "dirty": False},
            "dataset": {"sha256": "dataset"},
            "runtime": {"model": "model", "model_revision": "revision"},
            "request": {
                "max_new_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
            },
            "user_metadata": {
                "hardware": {"class": "test"},
                "software": {"gpu_driver": "not-applicable"},
            },
        }
    }


def candidate_summary(*, experiment_id: str = "candidate") -> dict[str, object]:
    value = summary()
    value["manifest"]["experiment_id"] = experiment_id
    value["manifest"]["runtime"]["backend"] = "candidate"
    value["runs"] = [
        {
            "concurrency": 1,
            "failure_rate": 0.0,
            "ttft_ms_p95": 20.0,
            "latency_ms_p95": 100.0,
            "request_throughput_rps": 10.0,
            "output_throughput_tokens_per_s": 500.0,
        },
        {
            "concurrency": 8,
            "failure_rate": 0.01,
            "ttft_ms_p95": 80.0,
            "latency_ms_p95": 300.0,
            "request_throughput_rps": 50.0,
            "output_throughput_tokens_per_s": 2_500.0,
        },
    ]
    return value


def reference_summary(*, experiment_id: str = "reference") -> dict[str, object]:
    value = summary()
    value["manifest"]["experiment_id"] = experiment_id
    value["manifest"]["runtime"]["backend"] = "reference"
    return value


def policy(**performance: object) -> EvaluationPolicy:
    defaults: dict[str, object] = {
        "target_concurrency": [8],
        "max_failure_rate": 0.02,
        "max_ttft_ms_p95": 100,
        "max_latency_ms_p95": 500,
        "min_output_throughput_tokens_per_s": 2_000,
    }
    defaults.update(performance)
    return EvaluationPolicy.model_validate(
        {
            "name": "test-policy",
            "performance": defaults,
            "equivalence": {
                "min_exact_match_rate": 1.0,
                "min_concurrency_stability_rate": 1.0,
            },
        }
    )


def with_experiment(rows: list[dict[str, object]], experiment_id: str) -> list[dict[str, object]]:
    for item in rows:
        item["experiment_id"] = experiment_id
    return rows


def test_performance_policy_passes_target_concurrency() -> None:
    result = evaluate_performance(candidate_summary(), policy().performance)

    assert result["passed"] is True
    assert result["target_concurrency"] == [8]
    assert all(check["passed"] for check in result["by_concurrency"][0]["checks"])


def test_missing_metric_fails_instead_of_being_ignored() -> None:
    candidate = candidate_summary()
    candidate["runs"][1]["output_throughput_tokens_per_s"] = None

    result = evaluate_performance(candidate, policy().performance)

    assert result["passed"] is False
    throughput = next(
        check
        for check in result["by_concurrency"][0]["checks"]
        if check["metric"] == "output_throughput_tokens_per_s"
    )
    assert throughput["observed"] is None
    assert throughput["passed"] is False


def test_policy_rejects_unknown_fields_and_empty_performance() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvaluationPolicy.model_validate(
            {"name": "bad", "performance": {"max_ttft_typo": 100}}
        )
    with pytest.raises(ValidationError, match="at least one threshold"):
        EvaluationPolicy.model_validate({"name": "empty", "performance": {}})


def test_committed_example_policy_is_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]

    loaded = load_policy(project_root / "policies" / "deterministic-serving.example.json")

    assert loaded.name == "example-deterministic-serving-policy"
    assert loaded.performance.target_concurrency == [8]


def test_report_passes_only_when_all_sections_pass() -> None:
    reference = with_experiment([row("prompt", "same", concurrency=1)], "reference")
    candidate = with_experiment([row("prompt", "same", concurrency=8)], "candidate")

    report = build_evaluation_report(
        reference_rows=reference,
        candidate_rows=candidate,
        reference_summary=reference_summary(),
        candidate_summary=candidate_summary(),
        policy=policy(),
    )

    assert report["controls"]["passed"] is True
    assert report["artifact_consistency"]["passed"] is True
    assert report["performance"]["passed"] is True
    assert report["equivalence"]["passed"] is True
    assert report["deployment_eligibility"]["eligible"] is True
    assert report["deployment_eligibility"]["failed_sections"] == []


def test_control_mismatch_makes_report_ineligible() -> None:
    reference = with_experiment([row("prompt", "same", concurrency=1)], "reference")
    candidate = with_experiment([row("prompt", "same", concurrency=8)], "candidate")
    mismatched = candidate_summary()
    mismatched["manifest"]["runtime"]["model"] = "other-model"

    report = build_evaluation_report(
        reference_rows=reference,
        candidate_rows=candidate,
        reference_summary=reference_summary(),
        candidate_summary=mismatched,
        policy=policy(),
    )

    assert report["deployment_eligibility"]["eligible"] is False
    assert report["deployment_eligibility"]["failed_sections"] == ["controls"]


def test_equivalence_scope_is_independent_from_performance_scope() -> None:
    reference = with_experiment([row("prompt", "same", concurrency=1)], "reference")
    candidate = with_experiment(
        [
            row("prompt", "changed", concurrency=1),
            row("prompt", "same", concurrency=8),
        ],
        "candidate",
    )

    report = build_evaluation_report(
        reference_rows=reference,
        candidate_rows=candidate,
        reference_summary=reference_summary(),
        candidate_summary=candidate_summary(),
        policy=policy(),
    )

    assert report["performance"]["target_concurrency"] == [8]
    assert report["performance"]["passed"] is True
    assert report["equivalence"]["target_concurrency"] == [1, 8]
    assert report["equivalence"]["passed"] is False
    assert report["deployment_eligibility"]["eligible"] is False


def test_evaluate_files_records_every_input_digest(tmp_path: Path) -> None:
    files = {
        "reference": tmp_path / "reference.jsonl",
        "reference_summary": tmp_path / "reference.summary.json",
        "candidate": tmp_path / "candidate.jsonl",
        "candidate_summary": tmp_path / "candidate.summary.json",
        "policy": tmp_path / "policy.json",
    }
    reference = with_experiment([row("prompt", "same", concurrency=1)], "reference")
    candidate = with_experiment([row("prompt", "same", concurrency=8)], "candidate")
    files["reference"].write_text(json.dumps(reference[0]) + "\n", encoding="utf-8")
    files["candidate"].write_text(json.dumps(candidate[0]) + "\n", encoding="utf-8")
    files["reference_summary"].write_text(
        json.dumps(reference_summary()), encoding="utf-8"
    )
    files["candidate_summary"].write_text(
        json.dumps(candidate_summary()), encoding="utf-8"
    )
    files["policy"].write_text(policy().model_dump_json(), encoding="utf-8")

    report = evaluate_files(
        reference=files["reference"],
        reference_summary=files["reference_summary"],
        candidate=files["candidate"],
        candidate_summary=files["candidate_summary"],
        policy_path=files["policy"],
    )

    assert report["deployment_eligibility"]["eligible"] is True
    assert set(report["artifacts"]) == {
        "reference_results",
        "reference_summary",
        "candidate_results",
        "candidate_summary",
        "policy",
    }
    assert all(len(artifact["sha256"]) == 64 for artifact in report["artifacts"].values())


def test_report_fails_when_raw_rows_do_not_match_summary() -> None:
    reference = with_experiment([row("prompt", "same", concurrency=1)], "wrong-id")
    candidate = with_experiment([row("prompt", "same", concurrency=8)], "candidate")

    report = build_evaluation_report(
        reference_rows=reference,
        candidate_rows=candidate,
        reference_summary=reference_summary(),
        candidate_summary=candidate_summary(),
        policy=policy(),
    )

    assert report["artifact_consistency"]["passed"] is False
    assert report["deployment_eligibility"]["failed_sections"] == [
        "artifact_consistency"
    ]


def test_cli_accepts_report_only_mode() -> None:
    args = parse_args(
        [
            "--reference",
            "reference.jsonl",
            "--reference-summary",
            "reference.summary.json",
            "--candidate",
            "candidate.jsonl",
            "--candidate-summary",
            "candidate.summary.json",
            "--policy",
            "policy.json",
            "--output",
            "report.json",
            "--report-only",
        ]
    )

    assert args.report_only is True
