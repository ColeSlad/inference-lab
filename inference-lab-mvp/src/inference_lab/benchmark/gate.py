import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from inference_lab.benchmark.evaluation import (
    compare_controls,
    evaluate_equivalence,
    load_json_object,
    load_jsonl,
)
from inference_lab.benchmark.runner import sha256_file

EVALUATION_REPORT_SCHEMA_VERSION = 1


class PerformancePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_concurrency: list[int] | None = None
    max_failure_rate: float | None = Field(default=None, ge=0, le=1)
    max_ttft_ms_p95: float | None = Field(default=None, ge=0)
    max_latency_ms_p95: float | None = Field(default=None, ge=0)
    min_request_throughput_rps: float | None = Field(default=None, ge=0)
    min_output_throughput_tokens_per_s: float | None = Field(default=None, ge=0)

    @field_validator("target_concurrency")
    @classmethod
    def validate_target_concurrency(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value or any(item < 1 for item in value):
            raise ValueError("target_concurrency must contain positive integers")
        if len(value) != len(set(value)):
            raise ValueError("target_concurrency must not contain duplicates")
        return value

    @model_validator(mode="after")
    def require_threshold(self) -> "PerformancePolicy":
        thresholds = (
            self.max_failure_rate,
            self.max_ttft_ms_p95,
            self.max_latency_ms_p95,
            self.min_request_throughput_rps,
            self.min_output_throughput_tokens_per_s,
        )
        if all(value is None for value in thresholds):
            raise ValueError("performance must define at least one threshold")
        return self


class EquivalencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_concurrency: list[int] | None = None
    min_exact_match_rate: float = Field(default=1.0, ge=0, le=1)
    min_matching_prefix_character_ratio: float | None = Field(default=None, ge=0, le=1)
    min_concurrency_stability_rate: float = Field(default=1.0, ge=0, le=1)
    mismatch_limit: int = Field(default=20, ge=0, le=1_000)

    @field_validator("target_concurrency")
    @classmethod
    def validate_target_concurrency(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value or any(item < 1 for item in value):
            raise ValueError("target_concurrency must contain positive integers")
        if len(value) != len(set(value)):
            raise ValueError("target_concurrency must not contain duplicates")
        return value


class EvaluationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    performance: PerformancePolicy
    equivalence: EquivalencePolicy = Field(default_factory=EquivalencePolicy)


PERFORMANCE_THRESHOLDS = (
    ("failure_rate", "max_failure_rate", "<="),
    ("ttft_ms_p95", "max_ttft_ms_p95", "<="),
    ("latency_ms_p95", "max_latency_ms_p95", "<="),
    ("request_throughput_rps", "min_request_throughput_rps", ">="),
    (
        "output_throughput_tokens_per_s",
        "min_output_throughput_tokens_per_s",
        ">=",
    ),
)


def load_policy(path: Path) -> EvaluationPolicy:
    try:
        return EvaluationPolicy.model_validate(load_json_object(path))
    except ValidationError as exc:
        raise ValueError(f"Invalid evaluation policy {path}: {exc}") from exc


def evaluate_performance(
    candidate_summary: dict[str, Any], policy: PerformancePolicy
) -> dict[str, Any]:
    runs = candidate_summary.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Candidate summary has no aggregate runs")

    runs_by_concurrency: dict[int, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Candidate aggregate runs must be JSON objects")
        concurrency = run.get("concurrency")
        if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
            raise ValueError("Every candidate aggregate run needs a positive concurrency")
        if concurrency in runs_by_concurrency:
            raise ValueError(f"Duplicate aggregate run for concurrency {concurrency}")
        runs_by_concurrency[concurrency] = run

    target_concurrency = policy.target_concurrency or sorted(runs_by_concurrency)
    missing = sorted(set(target_concurrency) - set(runs_by_concurrency))
    if missing:
        raise ValueError(f"Candidate summary is missing target concurrency: {missing}")

    results = []
    for concurrency in target_concurrency:
        run = runs_by_concurrency[concurrency]
        checks = []
        for metric, policy_field, operator in PERFORMANCE_THRESHOLDS:
            threshold = getattr(policy, policy_field)
            if threshold is None:
                continue
            observed_value = run.get(metric)
            observed = (
                float(observed_value)
                if isinstance(observed_value, int | float)
                and not isinstance(observed_value, bool)
                else None
            )
            checks.append(_performance_check(metric, observed, operator, threshold))
        results.append(
            {
                "concurrency": concurrency,
                "passed": all(bool(check["passed"]) for check in checks),
                "checks": checks,
            }
        )

    return {
        "passed": all(bool(result["passed"]) for result in results),
        "target_concurrency": target_concurrency,
        "by_concurrency": results,
    }


def build_evaluation_report(
    *,
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    policy: EvaluationPolicy,
    artifacts: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    controls = compare_controls(reference_summary, candidate_summary)
    performance = evaluate_performance(candidate_summary, policy.performance)
    candidate_concurrency = {
        row.get("concurrency")
        for row in candidate_rows
        if isinstance(row.get("concurrency"), int)
        and not isinstance(row.get("concurrency"), bool)
    }
    equivalence_targets = policy.equivalence.target_concurrency or sorted(
        candidate_concurrency
    )
    missing_equivalence_targets = sorted(
        set(equivalence_targets) - candidate_concurrency
    )
    if missing_equivalence_targets:
        raise ValueError(
            "Candidate results are missing equivalence target concurrency: "
            f"{missing_equivalence_targets}"
        )
    target_concurrency = set(equivalence_targets)
    evaluated_candidate_rows = [
        row for row in candidate_rows if row.get("concurrency") in target_concurrency
    ]
    if not evaluated_candidate_rows:
        raise ValueError("Candidate results have no rows at the target concurrency")

    equivalence_policy = policy.equivalence
    equivalence = evaluate_equivalence(
        reference_rows,
        evaluated_candidate_rows,
        min_exact_match_rate=equivalence_policy.min_exact_match_rate,
        min_matching_prefix_character_ratio=(
            equivalence_policy.min_matching_prefix_character_ratio
        ),
        min_concurrency_stability_rate=(
            equivalence_policy.min_concurrency_stability_rate
        ),
        mismatch_limit=equivalence_policy.mismatch_limit,
    )
    equivalence["target_concurrency"] = equivalence_targets
    artifact_consistency = _artifact_consistency(
        reference_rows,
        candidate_rows,
        reference_summary,
        candidate_summary,
    )
    eligible = all(
        bool(section["passed"])
        for section in (controls, artifact_consistency, performance, equivalence)
    )
    failed_sections = [
        name
        for name, section in (
            ("controls", controls),
            ("artifact_consistency", artifact_consistency),
            ("performance", performance),
            ("equivalence", equivalence),
        )
        if not section["passed"]
    ]

    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "policy": policy.model_dump(mode="json"),
        "artifacts": artifacts or {},
        "reference": _summary_identity(reference_summary),
        "candidate": _summary_identity(candidate_summary),
        "controls": controls,
        "artifact_consistency": artifact_consistency,
        "performance": performance,
        "equivalence": equivalence,
        "deployment_eligibility": {
            "eligible": eligible,
            "failed_sections": failed_sections,
            "scope": "benchmark policy only; not a general production-readiness certification",
        },
    }


def evaluate_files(
    *,
    reference: Path,
    reference_summary: Path,
    candidate: Path,
    candidate_summary: Path,
    policy_path: Path,
) -> dict[str, Any]:
    paths = {
        "reference_results": reference,
        "reference_summary": reference_summary,
        "candidate_results": candidate,
        "candidate_summary": candidate_summary,
        "policy": policy_path,
    }
    artifacts = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    return build_evaluation_report(
        reference_rows=load_jsonl(reference),
        candidate_rows=load_jsonl(candidate),
        reference_summary=load_json_object(reference_summary),
        candidate_summary=load_json_object(candidate_summary),
        policy=load_policy(policy_path),
        artifacts=artifacts,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate an optimized backend on performance and deterministic equivalence."
    )
    parser.add_argument("--reference", type=Path, required=True, help="Reference raw JSONL")
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate raw JSONL")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write an ineligible report without returning a failing exit status.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        report = evaluate_files(
            reference=args.reference,
            reference_summary=args.reference_summary,
            candidate=args.candidate,
            candidate_summary=args.candidate_summary,
            policy_path=args.policy,
        )
    except ValueError as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eligible = bool(report["deployment_eligibility"]["eligible"])
    print(f"Wrote evaluation report to {args.output}")
    print(f"Deployment eligibility: {'PASS' if eligible else 'FAIL'}")
    if not eligible and not args.report_only:
        raise SystemExit(1)


def _performance_check(
    metric: str, observed: float | None, operator: str, threshold: float
) -> dict[str, Any]:
    if operator == "<=":
        passed = observed is not None and observed <= threshold
    elif operator == ">=":
        passed = observed is not None and observed >= threshold
    else:
        raise ValueError(f"Unsupported performance operator: {operator}")
    return {
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": passed,
    }


def _artifact_consistency(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    reference_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        _experiment_id_check("reference", reference_rows, reference_summary),
        _experiment_id_check("candidate", candidate_rows, candidate_summary),
    ]
    return {"passed": all(bool(check["passed"]) for check in checks), "checks": checks}


def _experiment_id_check(
    label: str, rows: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    manifest = summary.get("manifest")
    summary_id = manifest.get("experiment_id") if isinstance(manifest, dict) else None
    row_ids = {row.get("experiment_id") for row in rows}
    return {
        "name": f"{label}_experiment_id",
        "summary": summary_id,
        "raw_values": sorted(str(value) for value in row_ids),
        "passed": summary_id is not None and row_ids == {summary_id},
    }


def _summary_identity(summary: dict[str, Any]) -> dict[str, Any]:
    manifest = summary.get("manifest")
    if not isinstance(manifest, dict):
        return {}
    runtime = manifest.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    return {
        "experiment_id": manifest.get("experiment_id"),
        "backend": runtime.get("backend"),
        "model": runtime.get("model"),
        "model_revision": runtime.get("model_revision"),
        "model_dtype": runtime.get("model_dtype"),
    }


if __name__ == "__main__":
    main()
