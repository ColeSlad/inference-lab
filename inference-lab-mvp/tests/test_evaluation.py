import hashlib

import pytest

from inference_lab.benchmark.evaluation import compare_controls, evaluate_equivalence


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row(
    prompt: str,
    output: str,
    *,
    concurrency: int,
    repetition: int = 1,
    evidence: str = "hash",
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "ok",
        "prompt_sha256": digest(prompt),
        "concurrency": concurrency,
        "repetition": repetition,
        "request_index": 0,
        "output_sha256": digest(output),
    }
    if evidence == "text":
        result["output_text"] = output
    return result


def summary(*, model: str = "model", temperature: float = 0.0) -> dict[str, object]:
    return {
        "manifest": {
            "git": {"commit": "commit", "dirty": False},
            "dataset": {"sha256": "dataset"},
            "runtime": {"model": model, "model_revision": "revision"},
            "request": {
                "max_new_tokens": 64,
                "temperature": temperature,
                "top_p": 1.0,
                "seed": 42,
            },
            "user_metadata": {
                "hardware": {"class": "test"},
                "software": {"gpu_driver": "not-applicable"},
            },
        }
    }


def test_hash_evidence_passes_exact_and_concurrency_stability() -> None:
    reference = [row("one", "same", concurrency=1), row("two", "other", concurrency=1)]
    candidate = [
        row("one", "same", concurrency=1),
        row("two", "other", concurrency=1),
        row("one", "same", concurrency=8),
        row("two", "other", concurrency=8),
    ]

    result = evaluate_equivalence(reference, candidate)

    assert result["passed"] is True
    assert result["exact_match_rate"] == 1.0
    assert result["concurrency_stability_rate"] == 1.0
    assert result["matching_prefix_character_ratio_mean"] is None
    assert [item["exact_match_rate"] for item in result["by_concurrency"]] == [1.0, 1.0]


def test_text_evidence_reports_divergence_without_copying_output() -> None:
    reference = [row("prompt", "abcdef", concurrency=1, evidence="text")]
    candidate = [row("prompt", "abcXYZ", concurrency=4, evidence="text")]

    result = evaluate_equivalence(
        reference,
        candidate,
        min_exact_match_rate=0.0,
        min_matching_prefix_character_ratio=0.5,
    )

    assert result["passed"] is True
    assert result["exact_match_rate"] == 0.0
    assert result["matching_prefix_character_ratio_mean"] == 0.5
    assert result["mismatches"] == [
        {
            "prompt_sha256": digest("prompt"),
            "concurrency": 4,
            "repetition": 1,
            "request_index": 0,
            "reference_output_sha256": digest("abcdef"),
            "candidate_output_sha256": digest("abcXYZ"),
            "reference_output_chars": 6,
            "candidate_output_chars": 6,
            "first_divergent_character": 3,
            "matching_prefix_character_ratio": 0.5,
        }
    ]
    assert "abcdef" not in str(result)
    assert "abcXYZ" not in str(result)


def test_errors_and_unstable_outputs_fail_equivalence() -> None:
    reference = [row("prompt", "reference", concurrency=1)]
    candidate = [
        row("prompt", "reference", concurrency=1),
        row("prompt", "changed", concurrency=8),
        {
            "status": "error",
            "prompt_sha256": digest("prompt"),
            "concurrency": 8,
        },
    ]

    result = evaluate_equivalence(reference, candidate)

    assert result["passed"] is False
    assert result["attempted_requests"] == 3
    assert result["exact_match_rate"] == pytest.approx(1 / 3)
    assert result["concurrency_stability_rate"] == 0.0


def test_reference_errors_invalidate_reference_stability() -> None:
    reference = [
        row("prompt", "reference", concurrency=1),
        {
            "status": "error",
            "prompt_sha256": digest("prompt"),
            "concurrency": 1,
        },
    ]
    candidate = [row("prompt", "reference", concurrency=1)]

    result = evaluate_equivalence(reference, candidate)

    assert result["passed"] is False
    assert result["reference_stability_rate"] == 0.0
    assert result["comparable_requests"] == 0


def test_missing_candidate_evidence_fails_closed_even_with_zero_thresholds() -> None:
    reference = [row("prompt", "reference", concurrency=1)]
    candidate = [
        {
            "status": "ok",
            "prompt_sha256": digest("prompt"),
            "concurrency": 1,
        }
    ]

    result = evaluate_equivalence(
        reference,
        candidate,
        min_exact_match_rate=0.0,
        min_concurrency_stability_rate=0.0,
    )

    assert result["passed"] is False
    assert result["candidate_evidence_coverage_rate"] == 0.0


def test_candidate_prompts_require_stable_reference_coverage() -> None:
    reference = [row("known", "reference", concurrency=1)]
    candidate = [row("unknown", "candidate", concurrency=1)]

    result = evaluate_equivalence(
        reference,
        candidate,
        min_exact_match_rate=0.0,
    )

    assert result["passed"] is False
    assert result["reference_prompt_coverage_rate"] == 0.0


def test_output_hash_must_match_captured_text() -> None:
    reference = [row("prompt", "reference", concurrency=1)]
    candidate = [row("prompt", "candidate", concurrency=1, evidence="text")]
    candidate[0]["output_sha256"] = digest("tampered")

    with pytest.raises(ValueError, match="does not match"):
        evaluate_equivalence(reference, candidate)


def test_control_comparison_requires_same_deterministic_inputs() -> None:
    controls = compare_controls(summary(), summary(model="different", temperature=0.5))

    assert controls["passed"] is False
    failed = {check["name"] for check in controls["checks"] if not check["passed"]}
    assert failed == {"runtime.model", "request.temperature", "deterministic_decoding"}


def test_control_comparison_requires_same_hardware_and_clean_git() -> None:
    candidate = summary()
    candidate["manifest"]["git"]["dirty"] = True
    candidate["manifest"]["user_metadata"]["hardware"] = {"class": "different"}

    controls = compare_controls(summary(), candidate)

    failed = {check["name"] for check in controls["checks"] if not check["passed"]}
    assert failed == {"user_metadata.hardware", "candidate_git_clean"}
