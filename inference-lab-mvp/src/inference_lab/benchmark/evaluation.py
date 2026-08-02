import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CONTROL_FIELDS = (
    ("dataset.sha256", ("dataset", "sha256")),
    ("runtime.model", ("runtime", "model")),
    ("runtime.model_revision", ("runtime", "model_revision")),
    ("request.max_new_tokens", ("request", "max_new_tokens")),
    ("request.temperature", ("request", "temperature")),
    ("request.top_p", ("request", "top_p")),
    ("request.seed", ("request", "seed")),
)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object on {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No result rows found in {path}")
    return rows


def compare_controls(
    reference_summary: dict[str, Any], candidate_summary: dict[str, Any]
) -> dict[str, Any]:
    reference_manifest = _manifest(reference_summary, "reference")
    candidate_manifest = _manifest(candidate_summary, "candidate")
    checks = []
    for name, path in CONTROL_FIELDS:
        reference_value = _nested(reference_manifest, path)
        candidate_value = _nested(candidate_manifest, path)
        checks.append(
            {
                "name": name,
                "reference": reference_value,
                "candidate": candidate_value,
                "passed": reference_value is not None and reference_value == candidate_value,
            }
        )

    temperature = _nested(candidate_manifest, ("request", "temperature"))
    checks.append(
        {
            "name": "deterministic_decoding",
            "expected": "temperature == 0",
            "observed": temperature,
            "passed": temperature == 0,
        }
    )
    return {"passed": all(bool(check["passed"]) for check in checks), "checks": checks}


def evaluate_equivalence(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    min_exact_match_rate: float = 1.0,
    min_matching_prefix_character_ratio: float | None = None,
    min_concurrency_stability_rate: float = 1.0,
    mismatch_limit: int = 20,
) -> dict[str, Any]:
    _validate_rate("min_exact_match_rate", min_exact_match_rate)
    _validate_rate("min_concurrency_stability_rate", min_concurrency_stability_rate)
    if min_matching_prefix_character_ratio is not None:
        _validate_rate(
            "min_matching_prefix_character_ratio",
            min_matching_prefix_character_ratio,
        )
    if mismatch_limit < 0:
        raise ValueError("mismatch_limit must be non-negative")

    reference_by_prompt = _group_by_prompt(reference_rows)
    candidate_by_prompt = _group_by_prompt(candidate_rows)
    reference_digests = {
        prompt_sha256: {
            _output_digest(row) if row.get("status") == "ok" else None for row in rows
        }
        for prompt_sha256, rows in reference_by_prompt.items()
    }
    stable_reference_digests = {
        prompt_sha256: next(iter(digests))
        for prompt_sha256, digests in reference_digests.items()
        if len(digests) == 1 and None not in digests
    }
    reference_prompt_count = len(reference_by_prompt)
    reference_stability_rate = (
        len(stable_reference_digests) / reference_prompt_count
        if reference_prompt_count
        else 0.0
    )

    reference_text = _stable_reference_text(reference_by_prompt, stable_reference_digests)
    candidate_attempted = len(candidate_rows)
    exact_matches = 0
    comparable = 0
    prefix_ratios: list[float] = []
    mismatches: list[dict[str, Any]] = []
    by_concurrency_counts: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"attempted_requests": 0, "comparable_requests": 0, "exact_matches": 0}
    )

    for row in candidate_rows:
        prompt_sha256 = _prompt_sha256(row)
        concurrency = _concurrency(row)
        counts = by_concurrency_counts[concurrency]
        counts["attempted_requests"] += 1
        candidate_digest = _output_digest(row)
        expected_digest = stable_reference_digests.get(prompt_sha256)
        is_comparable = row.get("status") == "ok" and candidate_digest is not None
        if is_comparable and expected_digest is not None:
            comparable += 1
            counts["comparable_requests"] += 1
            if candidate_digest == expected_digest:
                exact_matches += 1
                counts["exact_matches"] += 1
            elif len(mismatches) < mismatch_limit:
                mismatches.append(
                    _mismatch_record(
                        row,
                        expected_digest=expected_digest,
                        candidate_digest=candidate_digest,
                        expected_text=reference_text.get(prompt_sha256),
                    )
                )

        expected_text = reference_text.get(prompt_sha256)
        candidate_text = row.get("output_text")
        if row.get("status") == "ok" and isinstance(expected_text, str) and isinstance(
            candidate_text, str
        ):
            prefix_ratios.append(_matching_prefix_character_ratio(expected_text, candidate_text))

    exact_match_rate = exact_matches / candidate_attempted if candidate_attempted else 0.0
    prefix_evidence_complete = len(prefix_ratios) == candidate_attempted
    mean_prefix_ratio = (
        sum(prefix_ratios) / len(prefix_ratios)
        if prefix_evidence_complete and prefix_ratios
        else None
    )

    stable_candidate_prompts = 0
    for rows in candidate_by_prompt.values():
        digests = {_output_digest(row) for row in rows}
        if (
            all(row.get("status") == "ok" for row in rows)
            and None not in digests
            and len(digests) == 1
        ):
            stable_candidate_prompts += 1
    candidate_prompt_count = len(candidate_by_prompt)
    concurrency_stability_rate = (
        stable_candidate_prompts / candidate_prompt_count if candidate_prompt_count else 0.0
    )

    by_concurrency = []
    for concurrency in sorted(by_concurrency_counts):
        counts = by_concurrency_counts[concurrency]
        attempted = int(counts["attempted_requests"])
        matches = int(counts["exact_matches"])
        by_concurrency.append(
            {
                "concurrency": concurrency,
                **counts,
                "exact_match_rate": matches / attempted if attempted else 0.0,
            }
        )

    checks = [
        _threshold_check(
            "reference_stability_rate", reference_stability_rate, ">=", 1.0
        ),
        _threshold_check(
            "exact_match_rate", exact_match_rate, ">=", min_exact_match_rate
        ),
        _threshold_check(
            "concurrency_stability_rate",
            concurrency_stability_rate,
            ">=",
            min_concurrency_stability_rate,
        ),
    ]
    if min_matching_prefix_character_ratio is not None:
        checks.append(
            _threshold_check(
                "matching_prefix_character_ratio_mean",
                mean_prefix_ratio,
                ">=",
                min_matching_prefix_character_ratio,
            )
        )

    return {
        "passed": all(bool(check["passed"]) for check in checks),
        "reference_prompt_count": reference_prompt_count,
        "reference_stable_prompt_count": len(stable_reference_digests),
        "reference_stability_rate": reference_stability_rate,
        "candidate_prompt_count": candidate_prompt_count,
        "candidate_stable_prompt_count": stable_candidate_prompts,
        "concurrency_stability_rate": concurrency_stability_rate,
        "attempted_requests": candidate_attempted,
        "comparable_requests": comparable,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "prefix_comparable_requests": len(prefix_ratios),
        "matching_prefix_character_ratio_mean": mean_prefix_ratio,
        "by_concurrency": by_concurrency,
        "checks": checks,
        "mismatches": mismatches,
    }


def _manifest(summary: dict[str, Any], label: str) -> dict[str, Any]:
    manifest = summary.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{label} summary has no manifest object")
    return manifest


def _nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _group_by_prompt(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_prompt_sha256(row)].append(row)
    return grouped


def _prompt_sha256(row: dict[str, Any]) -> str:
    value = row.get("prompt_sha256")
    if not isinstance(value, str) or not value:
        raise ValueError("Every result row must contain prompt_sha256")
    return value


def _concurrency(row: dict[str, Any]) -> int:
    value = row.get("concurrency")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("Every candidate row must contain a positive integer concurrency")
    return value


def _output_digest(row: dict[str, Any]) -> str | None:
    recorded = row.get("output_sha256")
    text = row.get("output_text")
    computed = hashlib.sha256(text.encode("utf-8")).hexdigest() if isinstance(text, str) else None
    if recorded is not None and not isinstance(recorded, str):
        raise ValueError("output_sha256 must be a string when present")
    if recorded is not None and computed is not None and recorded != computed:
        raise ValueError("output_sha256 does not match output_text")
    return recorded or computed


def _stable_reference_text(
    reference_by_prompt: dict[str, list[dict[str, Any]]],
    stable_reference_digests: dict[str, str],
) -> dict[str, str]:
    text_by_prompt: dict[str, str] = {}
    for prompt_sha256, expected_digest in stable_reference_digests.items():
        texts = {
            row["output_text"]
            for row in reference_by_prompt[prompt_sha256]
            if isinstance(row.get("output_text"), str)
        }
        if len(texts) == 1:
            text = next(iter(texts))
            if hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_digest:
                text_by_prompt[prompt_sha256] = text
    return text_by_prompt


def _matching_prefix_character_ratio(reference: str, candidate: str) -> float:
    prefix_length = 0
    for reference_character, candidate_character in zip(reference, candidate, strict=False):
        if reference_character != candidate_character:
            break
        prefix_length += 1
    denominator = max(len(reference), len(candidate))
    return prefix_length / denominator if denominator else 1.0


def _mismatch_record(
    row: dict[str, Any],
    *,
    expected_digest: str,
    candidate_digest: str,
    expected_text: str | None,
) -> dict[str, Any]:
    candidate_text = row.get("output_text")
    first_divergent_character: int | None = None
    prefix_ratio: float | None = None
    if isinstance(expected_text, str) and isinstance(candidate_text, str):
        shared = 0
        for reference_character, candidate_character in zip(
            expected_text, candidate_text, strict=False
        ):
            if reference_character != candidate_character:
                break
            shared += 1
        first_divergent_character = shared
        prefix_ratio = _matching_prefix_character_ratio(expected_text, candidate_text)

    return {
        "prompt_sha256": _prompt_sha256(row),
        "concurrency": _concurrency(row),
        "repetition": row.get("repetition"),
        "request_index": row.get("request_index"),
        "reference_output_sha256": expected_digest,
        "candidate_output_sha256": candidate_digest,
        "reference_output_chars": len(expected_text) if expected_text is not None else None,
        "candidate_output_chars": len(candidate_text) if isinstance(candidate_text, str) else None,
        "first_divergent_character": first_divergent_character,
        "matching_prefix_character_ratio": prefix_ratio,
    }


def _threshold_check(
    name: str, observed: float | None, operator: str, threshold: float
) -> dict[str, Any]:
    if operator != ">=":
        raise ValueError(f"Unsupported threshold operator: {operator}")
    return {
        "name": name,
        "operator": operator,
        "threshold": threshold,
        "observed": observed,
        "passed": observed is not None and observed >= threshold,
    }


def _validate_rate(name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
