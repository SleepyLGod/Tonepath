"""Data-driven selection benchmark checks for Tonepath."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path


CheckStatus = str
Scenario = dict[str, object]

CHECK_TYPES = {
    "no_high_vocalness_top_k",
    "max_stimulation_top_k",
    "prefer_low_vocalness_top_k",
    "no_duplicate_candidates",
    "required_confidence_top_k",
    "metadata_hygiene_warning",
}
STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def load_benchmark_scenarios(path: Path | None = None) -> list[Scenario]:
    """Load benchmark scenarios from a JSONL resource or explicit path."""

    source = path or benchmark_resource_path()
    scenarios: list[Scenario] = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Benchmark scenario line {index} must be a JSON object.")
        validate_scenario(payload, index=index)
        scenarios.append(payload)
    return scenarios


def benchmark_resource_path() -> Path:
    """Return the packaged selection benchmark resource path."""

    return Path(str(resources.files("tonepath").joinpath("resources", "selection_benchmark.jsonl")))


def scenario_from_prompt(prompt: str, limit: int) -> Scenario:
    """Return an ad hoc benchmark scenario for an explicit prompt."""

    return {
        "id": "ad_hoc",
        "lang": "unknown",
        "prompt": prompt,
        "limit": limit,
        "expected_intent": {},
        "checks": [],
    }


def validate_scenario(scenario: Scenario, index: int = 0) -> None:
    """Validate one benchmark scenario shape."""

    prefix = f"Benchmark scenario line {index}" if index else "Benchmark scenario"
    for key in ("id", "lang", "prompt", "limit", "expected_intent", "checks"):
        if key not in scenario:
            raise RuntimeError(f"{prefix} missing required field: {key}")
    if not isinstance(scenario["id"], str) or not scenario["id"]:
        raise RuntimeError(f"{prefix} id must be a non-empty string.")
    if not isinstance(scenario["prompt"], str) or not scenario["prompt"]:
        raise RuntimeError(f"{prefix} prompt must be a non-empty string.")
    if not isinstance(scenario["limit"], int) or int(scenario["limit"]) <= 0:
        raise RuntimeError(f"{prefix} limit must be a positive integer.")
    if not isinstance(scenario["expected_intent"], dict):
        raise RuntimeError(f"{prefix} expected_intent must be an object.")
    checks = scenario["checks"]
    if not isinstance(checks, list):
        raise RuntimeError(f"{prefix} checks must be a list.")
    for check in checks:
        if not isinstance(check, dict):
            raise RuntimeError(f"{prefix} check entries must be objects.")
        check_type = check.get("type")
        if check_type not in CHECK_TYPES:
            raise RuntimeError(f"{prefix} has unsupported check type: {check_type}")


def evaluate_benchmark_scenario(
    scenario: Scenario,
    actual_intent: dict[str, object],
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    """Return benchmark check results for one scenario and selected candidates."""

    check_results = [intent_check_result(scenario, actual_intent)]
    raw_checks = scenario.get("checks", [])
    if not isinstance(raw_checks, list):
        raise RuntimeError("Benchmark scenario checks must be a list.")
    for check in raw_checks:
        if not isinstance(check, dict):
            raise RuntimeError("Benchmark check entries must be objects.")
        check_results.append(run_check(check, candidates))
    return {
        "scenario_id": scenario["id"],
        "lang": scenario["lang"],
        "expected_intent": scenario["expected_intent"],
        "actual_intent": actual_intent,
        "result": aggregate_result(check_results),
        "checks": check_results,
    }


def intent_check_result(scenario: Scenario, actual_intent: dict[str, object]) -> dict[str, object]:
    """Return whether parsed intent matches the scenario expectation."""

    expected = scenario.get("expected_intent", {})
    if not expected:
        return check_result("expected_intent", "pass", "No expected intent declared.", [])
    if not isinstance(expected, dict):
        raise RuntimeError("expected_intent must be an object.")
    mismatches = [key for key, value in expected.items() if actual_intent.get(key) != value]
    if mismatches:
        return check_result("expected_intent", "fail", f"Intent mismatch: {', '.join(mismatches)}.", [])
    return check_result("expected_intent", "pass", "Parsed intent matches scenario.", [])


def run_check(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Run one configured benchmark check."""

    check_type = str(check["type"])
    if check_type == "no_high_vocalness_top_k":
        return no_high_vocalness_top_k(check, candidates)
    if check_type == "max_stimulation_top_k":
        return max_stimulation_top_k(check, candidates)
    if check_type == "prefer_low_vocalness_top_k":
        return prefer_low_vocalness_top_k(check, candidates)
    if check_type == "no_duplicate_candidates":
        return no_duplicate_candidates(check, candidates)
    if check_type == "required_confidence_top_k":
        return required_confidence_top_k(check, candidates)
    if check_type == "metadata_hygiene_warning":
        return metadata_hygiene_warning(check, candidates)
    raise RuntimeError(f"Unsupported benchmark check type: {check_type}")


def no_high_vocalness_top_k(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Check that top-k candidates are not vocal-heavy."""

    k = check_int(check, "k", default=3)
    max_vocalness = check_float(check, "max_vocalness", default=0.65)
    affected = [
        rank
        for rank, row in top_k(candidates, k)
        if (value := feature_float(row, "vocalness")) is not None and value >= max_vocalness
    ]
    return violation_result(check, affected, f"Top {k} has vocalness >= {max_vocalness}.")


def max_stimulation_top_k(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Check that top-k candidates stay below stimulation thresholds."""

    k = check_int(check, "k", default=3)
    max_bpm = optional_check_float(check, "max_bpm")
    max_energy = optional_check_float(check, "max_energy")
    max_loudness = optional_check_float(check, "max_loudness")
    affected: list[int] = []
    for rank, row in top_k(candidates, k):
        if max_bpm is not None and (bpm := feature_float(row, "bpm")) is not None and bpm >= max_bpm:
            affected.append(rank)
            continue
        if max_energy is not None and (energy := feature_float(row, "energy")) is not None and energy >= max_energy:
            affected.append(rank)
            continue
        if max_loudness is not None and (loudness := feature_float(row, "loudness")) is not None and loudness >= max_loudness:
            affected.append(rank)
    return violation_result(check, affected, f"Top {k} exceeds stimulation thresholds.")


def prefer_low_vocalness_top_k(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Check that enough top-k candidates have low vocalness."""

    k = check_int(check, "k", default=6)
    max_vocalness = check_float(check, "max_vocalness", default=0.35)
    min_count = check_int(check, "min_count", default=max(1, k // 2))
    low_count = sum(
        1
        for _, row in top_k(candidates, k)
        if (value := feature_float(row, "vocalness")) is not None and value <= max_vocalness
    )
    if low_count >= min_count:
        return check_result(str(check["type"]), "pass", f"Top {k} has {low_count} low-vocalness candidate(s).", [])
    affected = [rank for rank, _ in top_k(candidates, k)]
    return violation_result(check, affected, f"Top {k} has only {low_count}/{min_count} low-vocalness candidate(s).")


def no_duplicate_candidates(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Check that candidates do not repeat canonical tracks."""

    seen: set[tuple[object, ...]] = set()
    affected: list[int] = []
    for rank, row in enumerate(candidates, start=1):
        track = row.get("track")
        if not isinstance(track, dict):
            continue
        raw_key = track.get("canonical_key")
        if not isinstance(raw_key, list):
            continue
        key = tuple(raw_key)
        if key in seen:
            affected.append(rank)
        seen.add(key)
    return violation_result(check, affected, "Duplicate canonical candidates found.")


def required_confidence_top_k(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Check that top-k candidates meet a minimum confidence level."""

    k = check_int(check, "k", default=3)
    min_confidence = str(check.get("min_confidence", "medium"))
    min_value = CONFIDENCE_ORDER.get(min_confidence, 1)
    require_source = bool(check.get("require_feature_source", False))
    affected: list[int] = []
    for rank, row in top_k(candidates, k):
        confidence = str(row.get("confidence", "low"))
        if CONFIDENCE_ORDER.get(confidence, 0) < min_value:
            affected.append(rank)
            continue
        features = row.get("features")
        if require_source and isinstance(features, dict) and features.get("source") is None:
            affected.append(rank)
    return violation_result(check, affected, f"Top {k} does not meet confidence requirement.")


def metadata_hygiene_warning(check: dict[str, object], candidates: list[dict[str, object]]) -> dict[str, object]:
    """Warn when displayed candidates have dirty raw metadata."""

    affected: list[int] = []
    for rank, row in enumerate(candidates, start=1):
        track = row.get("track")
        if isinstance(track, dict) and track.get("metadata_issues"):
            affected.append(rank)
    if affected:
        return check_result(str(check["type"]), "warn", "Candidate metadata needs cleanup.", affected)
    return check_result(str(check["type"]), "pass", "Candidate metadata is clean.", [])


def violation_result(check: dict[str, object], affected_ranks: list[int], message: str) -> dict[str, object]:
    """Return pass or configured violation status for affected ranks."""

    check_type = str(check["type"])
    if not affected_ranks:
        return check_result(check_type, "pass", "Check passed.", [])
    return check_result(check_type, check_level(check), message, affected_ranks)


def check_result(check_type: str, status: CheckStatus, message: str, affected_ranks: list[int]) -> dict[str, object]:
    """Return one JSON-safe benchmark check result."""

    return {
        "type": check_type,
        "status": status,
        "message": message,
        "affected_ranks": affected_ranks,
    }


def aggregate_result(checks: list[dict[str, object]]) -> str:
    """Return PASS, WARN, or FAIL for a list of check results."""

    worst = max((STATUS_ORDER.get(str(check.get("status")), 0) for check in checks), default=0)
    if worst >= STATUS_ORDER["fail"]:
        return "FAIL"
    if worst >= STATUS_ORDER["warn"]:
        return "WARN"
    return "PASS"


def top_k(candidates: list[dict[str, object]], k: int) -> list[tuple[int, dict[str, object]]]:
    """Return 1-based rank and row pairs for top-k candidates."""

    return list(enumerate(candidates[:k], start=1))


def feature_float(row: dict[str, object], field: str) -> float | None:
    """Return a numeric feature value from a candidate row."""

    features = row.get("features")
    if not isinstance(features, dict):
        return None
    value = features.get(field)
    if value is None:
        return None
    return float(value)


def check_level(check: dict[str, object]) -> CheckStatus:
    """Return the configured violation level for a check."""

    level = str(check.get("level", "fail")).lower()
    if level not in {"warn", "fail"}:
        raise RuntimeError("Benchmark check level must be warn or fail.")
    return level


def check_int(check: dict[str, object], key: str, default: int) -> int:
    """Return an integer check parameter."""

    value = check.get(key, default)
    if not isinstance(value, int):
        raise RuntimeError(f"Benchmark check field {key} must be an integer.")
    return value


def check_float(check: dict[str, object], key: str, default: float) -> float:
    """Return a float check parameter."""

    value = check.get(key, default)
    if not isinstance(value, int | float):
        raise RuntimeError(f"Benchmark check field {key} must be numeric.")
    return float(value)


def optional_check_float(check: dict[str, object], key: str) -> float | None:
    """Return an optional float check parameter."""

    if key not in check:
        return None
    return check_float(check, key, default=0.0)
