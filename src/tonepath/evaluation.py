"""Read-only selection evaluation helpers."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from importlib import resources
from pathlib import Path

from tonepath import config
from tonepath.benchmark import evaluate_benchmark_scenario, load_benchmark_scenarios, scenario_from_prompt
from tonepath.db import TonepathStore
from tonepath.display import (
    canonical_track_key,
    dirty_metadata_issues_from_values,
    display_artist,
    display_label,
    display_title,
    normalize_key_part,
)
from tonepath.models import CandidateScore, SessionPlan, TrackFeatures
from tonepath.planner import parse_request, plan_session, request_constraints
from tonepath.selector import select_path


def evaluate_selection(store: TonepathStore, prompt: str, limit: int, profile_enabled: bool = True) -> dict[str, object]:
    """Return stable selection-evaluation rows without writing profile state."""

    plan = plan_session(prompt)
    return {
        "prompt": prompt,
        "profile_enabled": profile_enabled,
        "candidates": [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, limit, profile_enabled=profile_enabled)],
    }


def evaluate_audit(store: TonepathStore, prompt: str, limit: int) -> dict[str, object]:
    """Build a stable local audit evidence pack without profile writes."""

    plan = plan_session(prompt)
    rows = [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, limit)]
    annotate_red_flags(rows, no_vocals=plan.request.no_vocals)
    annotate_yellow_flags(rows, no_vocals=plan.request.no_vocals)
    run_id = uuid.uuid4().hex
    payload: dict[str, object] = {
        "run_id": run_id,
        "prompt": prompt,
        "source_state": plan.request.source_state,
        "target_state": plan.request.target_state,
        "duration_min": plan.request.duration_sec // 60,
        "constraints": request_constraints(plan.request),
        "candidates": rows,
    }
    audit_dir = audit_cache_dir(run_id)
    audit_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = audit_dir / "evidence.json"
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["cache_dir"] = str(audit_dir)
    payload["evidence_path"] = str(evidence_path)
    return payload


def run_codex_audit(evidence: dict[str, object], web: bool) -> dict[str, object]:
    """Run Codex against a local audit evidence pack and return structured output."""

    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("Codex CLI is not available on PATH. Install Codex or rerun without --codex.")
    evidence_path = evidence.get("evidence_path")
    cache_dir = evidence.get("cache_dir")
    if not isinstance(evidence_path, str) or not isinstance(cache_dir, str):
        raise RuntimeError("Audit evidence pack is missing cache paths.")
    result_path = Path(cache_dir) / "codex-result.json"
    command = [codex]
    if web:
        command.append("--search")
    command.extend(
        [
            "exec",
            "--sandbox",
            "read-only",
            "--cd",
            str(config.repo_root()),
            "--output-schema",
            str(codex_audit_schema_path()),
            "-o",
            str(result_path),
            "-",
        ]
    )
    try:
        subprocess.run(
            command,
            input=codex_prompt(evidence_path, web=web),
            text=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(codex_failure_message(exc)) from exc
    result = load_codex_audit_result(result_path)
    return {
        "evidence": evidence,
        "codex": result,
        "codex_result_path": str(result_path),
        "web_enabled": web,
    }


def evaluate_suite(store: TonepathStore, limit: int, prompts: tuple[str, ...] | None = None) -> list[dict[str, object]]:
    """Return product-oriented selection quality checks for multiple prompts."""

    payload: list[dict[str, object]] = []
    scenarios = [scenario_from_prompt(prompt, limit) for prompt in prompts] if prompts is not None else load_benchmark_scenarios()
    for scenario in scenarios:
        prompt = str(scenario["prompt"])
        plan = plan_session(prompt)
        scenario_limit = min(limit, int(scenario["limit"]))
        rows = [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, scenario_limit)]
        annotate_red_flags(rows, no_vocals=plan.request.no_vocals)
        annotate_yellow_flags(rows, no_vocals=plan.request.no_vocals)
        actual_intent = request_intent_payload(plan)
        benchmark = evaluate_benchmark_scenario(scenario, actual_intent, rows)
        payload.append(
            {
                "scenario_id": scenario["id"],
                "lang": scenario["lang"],
                "prompt": prompt,
                "source_state": plan.request.source_state,
                "target_state": plan.request.target_state,
                "duration_min": plan.request.duration_sec // 60,
                "constraints": request_constraints(plan.request),
                "expected_intent": benchmark["expected_intent"],
                "actual_intent": benchmark["actual_intent"],
                "result": benchmark["result"],
                "checks": benchmark["checks"],
                "red_flag_count": sum(len(row["red_flags"]) for row in rows),
                "yellow_flag_count": sum(len(row["yellow_flags"]) for row in rows),
                "dirty_metadata_count": candidate_dirty_metadata_count(rows),
                "duplicate_candidate_count": duplicate_candidate_count(rows),
                "candidates": rows,
            }
        )
    return payload


def evaluate_intent() -> dict[str, object]:
    """Evaluate deterministic prompt parsing against the packaged intent corpus."""

    cases = [intent_case_result(case) for case in load_intent_cases()]
    failures = [case for case in cases if not case["passed"]]
    return {
        "total": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "cases": cases,
    }


def load_intent_cases() -> list[dict[str, object]]:
    """Load packaged bilingual intent fixtures."""

    path = package_resource_path("resources", "intent_prompts.jsonl")
    cases: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError("Intent fixture rows must be JSON objects.")
            cases.append(payload)
    return cases


def intent_case_result(case: dict[str, object]) -> dict[str, object]:
    """Return expected and actual parser output for one intent fixture."""

    prompt = str(case["prompt"])
    request = parse_request(prompt)
    expected = {
        "source_state": case["source_state"],
        "target_state": case["target_state"],
        "duration_min": case["duration_min"],
        "no_vocals": case["no_vocals"],
        "quiet": case["quiet"],
    }
    actual = {
        "source_state": request.source_state,
        "target_state": request.target_state,
        "duration_min": request.duration_sec // 60,
        "no_vocals": request.no_vocals,
        "quiet": request.quiet,
    }
    return {
        "lang": case.get("lang", "unknown"),
        "prompt": prompt,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }


def evaluate_rerank(prompt: str) -> dict[str, object]:
    """Return an advisory rerank preview from the newest matching Codex audit."""

    latest = latest_codex_audit_for_prompt(prompt)
    if latest is None:
        return {
            "found": False,
            "prompt": prompt,
            "message": "No matching Codex audit result found for this prompt.",
        }
    evidence, codex, result_path = latest
    candidates = evidence.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("Audit evidence must contain a candidates list.")
    decisions = codex.get("decisions")
    if not isinstance(decisions, list):
        raise RuntimeError("Codex audit result must contain a decisions list.")
    decision_by_track = {int(item["track_id"]): item for item in decisions if isinstance(item, dict)}
    details = [rerank_detail(index, candidate, decision_by_track) for index, candidate in enumerate(candidates, start=1)]
    suggested_queue = suggested_rerank_queue(details)
    counts = {
        "keep": sum(1 for row in details if row["decision"] == "keep"),
        "demote": sum(1 for row in details if row["decision"] == "demote"),
        "reject": sum(1 for row in details if row["decision"] == "reject"),
        "not_audited": sum(1 for row in details if row["decision"] == "not_audited"),
    }
    return {
        "found": True,
        "prompt": prompt,
        "run_id": evidence.get("run_id", result_path.parent.name),
        "summary": codex.get("summary", "Codex rerank guidance available."),
        "counts": counts,
        "evidence_path": str(result_path.parent / "evidence.json"),
        "codex_result_path": str(result_path),
        "details": details,
        "suggested_queue": suggested_queue,
    }


def latest_codex_audit_for_prompt(prompt: str) -> tuple[dict[str, object], dict[str, object], Path] | None:
    """Load the newest Codex audit whose evidence prompt matches the requested prompt."""

    audit_root = config.ensure_data_dir() / "cache" / "audit"
    if not audit_root.exists():
        return None
    results = sorted(audit_root.glob("*/codex-result.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for result_path in results:
        evidence_path = result_path.parent / "evidence.json"
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(evidence, dict) or evidence.get("prompt") != prompt:
            continue
        return evidence, load_codex_audit_result(result_path), result_path
    return None


def rerank_detail(index: int, candidate: object, decisions: dict[int, dict[str, object]]) -> dict[str, object]:
    """Return one advisory rerank row for a candidate and optional Codex decision."""

    if not isinstance(candidate, dict):
        raise RuntimeError("Audit candidate must be an object.")
    track = candidate.get("track")
    if not isinstance(track, dict):
        raise RuntimeError("Audit candidate must contain a track object.")
    track_id = track.get("id")
    decision = decisions.get(int(track_id)) if isinstance(track_id, int) else None
    if decision is None:
        decision_name = "not_audited"
        fit_score = None
        risk_flags: list[object] = []
        reason = "No Codex decision for this candidate."
    else:
        decision_name = str(decision["decision"])
        fit_score = decision.get("fit_score")
        raw_flags = decision.get("risk_flags", [])
        risk_flags = raw_flags if isinstance(raw_flags, list) else []
        reason = str(decision.get("reason", ""))
    return {
        "original_rank": index,
        "phase": candidate.get("phase"),
        "track": track,
        "score": candidate.get("score"),
        "decision": decision_name,
        "fit_score": fit_score,
        "risk_flags": risk_flags,
        "reason": reason,
        "suggested_action": suggested_action(decision_name),
    }


def suggested_rerank_queue(details: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return keep/not-audited/demote rows while excluding rejected rows."""

    priority = {"keep": 0, "not_audited": 1, "demote": 2}
    queue = [row for row in details if row["decision"] != "reject"]
    ordered = sorted(queue, key=lambda row: (priority.get(str(row["decision"]), 1), int(row["original_rank"])))
    for rank, row in enumerate(ordered, start=1):
        row["suggested_rank"] = rank
    return ordered


def suggested_action(decision: str) -> str:
    """Return the human action implied by one audit decision."""

    if decision == "keep":
        return "keep"
    if decision == "demote":
        return "move later"
    if decision == "reject":
        return "remove from suggested queue"
    return "keep original position"


def eval_candidates(store: TonepathStore, plan: SessionPlan, limit: int, profile_enabled: bool = True) -> list[CandidateScore]:
    """Return a small balanced candidate set for evaluation output."""

    per_phase = max(1, math.ceil(limit / max(len(plan.phases), 1)))
    return select_path(store, plan, limit_per_phase=per_phase, profile_enabled=profile_enabled)[:limit]


def candidate_to_eval_row(store: TonepathStore, candidate: CandidateScore) -> dict[str, object]:
    """Convert one candidate into a stable read-only evaluation row."""

    features = store.get_features(candidate.track.id) if candidate.track.id is not None else None
    return {
        "phase": candidate.phase.label,
        "track": {
            "id": candidate.track.id,
            "title": candidate.track.title,
            "artist": candidate.track.artist,
            "album": candidate.track.album,
            "genre": candidate.track.genre,
            "format": candidate.track.format,
            "display_title": display_title(candidate.track),
            "display_artist": display_artist(candidate.track),
            "display_label": display_label(candidate.track),
            "canonical_key": list(canonical_track_key(candidate.track)),
            "metadata_issues": dirty_metadata_issues_from_values(candidate.track.title, candidate.track.artist),
        },
        "score": round(candidate.score, 3),
        "confidence": candidate.confidence,
        "features": features_to_eval_row(features),
        "reasons": list(candidate.reasons),
        "red_flags": [],
        "yellow_flags": [],
    }


def features_to_eval_row(features: TrackFeatures | None) -> dict[str, object]:
    """Convert stored feature values into JSON-safe evaluation fields."""

    if features is None:
        return {
            "source": None,
            "confidence": None,
            "energy": None,
            "loudness": None,
            "bpm": None,
            "vocalness": None,
        }
    return {
        "source": features.feature_source,
        "confidence": features.confidence,
        "energy": round(features.energy, 3) if features.energy is not None else None,
        "loudness": round(features.loudness, 2) if features.loudness is not None else None,
        "bpm": round(features.bpm, 1) if features.bpm is not None else None,
        "vocalness": round(features.vocalness, 3) if features.vocalness is not None else None,
    }


def candidate_dirty_metadata_count(rows: list[dict[str, object]]) -> int:
    """Return how many displayed candidates have dirty raw metadata."""

    count = 0
    for row in rows:
        track = row.get("track")
        if isinstance(track, dict) and track.get("metadata_issues"):
            count += 1
    return count


def duplicate_candidate_count(rows: list[dict[str, object]]) -> int:
    """Return duplicate displayed candidates beyond the first occurrence."""

    keys: list[tuple[str, str, int]] = []
    for row in rows:
        track = row.get("track")
        if not isinstance(track, dict):
            continue
        raw_key = track.get("canonical_key")
        if isinstance(raw_key, list) and len(raw_key) == 3:
            keys.append((str(raw_key[0]), str(raw_key[1]), int(raw_key[2])))
            continue
        keys.append(
            (
                normalize_key_part(str(track.get("display_title") or track.get("title") or "")),
                normalize_key_part(str(track.get("display_artist") or track.get("artist") or "")),
                -1,
            )
        )
    return duplicate_track_count_from_keys(keys)


def duplicate_track_count_from_keys(keys: list[tuple[str, str, int]]) -> int:
    """Return duplicate count from canonical keys."""

    counts: dict[tuple[str, str, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def annotate_red_flags(rows: list[dict[str, object]], no_vocals: bool) -> None:
    """Attach product-quality red flags to evaluation rows in place."""

    for row in rows:
        row["red_flags"] = candidate_red_flags(row, no_vocals=no_vocals)


def annotate_yellow_flags(rows: list[dict[str, object]], no_vocals: bool) -> None:
    """Attach review warnings to evaluation rows in place."""

    for row in rows:
        row["yellow_flags"] = candidate_yellow_flags(row, no_vocals=no_vocals)


def candidate_red_flags(row: dict[str, object], no_vocals: bool) -> list[str]:
    """Return product-quality red flags for one candidate."""

    features = row.get("features")
    if not isinstance(features, dict):
        return ["missing feature payload"]
    flags: list[str] = []
    vocalness = float_or_none(features.get("vocalness"))
    energy = float_or_none(features.get("energy"))
    loudness = float_or_none(features.get("loudness"))
    bpm = float_or_none(features.get("bpm"))
    confidence = row.get("confidence")
    phase = str(row.get("phase") or "")

    if no_vocals and vocalness is not None and vocalness >= 0.65:
        flags.append("high vocalness in no-vocals candidate")
    if confidence == "low" or features.get("source") is None:
        flags.append("low evidence candidate")
    if phase in {"decompress", "focus"}:
        if energy is not None and energy >= 0.75:
            flags.append("high energy in calm/focus candidate")
        if loudness is not None and loudness >= -8.0:
            flags.append("high loudness in calm/focus candidate")
        if bpm is not None and bpm >= 150.0:
            flags.append("high BPM in calm/focus candidate")
    return flags


def candidate_yellow_flags(row: dict[str, object], no_vocals: bool) -> list[str]:
    """Return non-fatal review warnings for one candidate."""

    features = row.get("features")
    if not isinstance(features, dict):
        return []
    flags: list[str] = []
    vocalness = float_or_none(features.get("vocalness"))
    energy = float_or_none(features.get("energy"))
    loudness = float_or_none(features.get("loudness"))
    bpm = float_or_none(features.get("bpm"))
    phase = str(row.get("phase") or "")
    calm_phase = phase in {"decompress", "focus", "soften", "settle", "calm"}

    if no_vocals and vocalness is not None and 0.35 <= vocalness < 0.65:
        flags.append("vocalness inconclusive for no-vocals")
    if calm_phase and bpm is not None and bpm >= 140.0:
        if vocalness is not None and vocalness < 0.35:
            flags.append("instrumental but potentially overstimulating")
        else:
            flags.append("high BPM in low-stim phase")
    if calm_phase and energy is not None and energy >= 0.68:
        flags.append("high energy for calm/focus")
    if calm_phase and loudness is not None and loudness >= -9.0:
        flags.append("high loudness for calm/focus")
    if calm_phase and vocalness is not None and vocalness >= 0.65:
        flags.append("vocal-heavy in low-stim prompt")
    return flags


def float_or_none(value: object) -> float | None:
    """Return a float for numeric evaluation values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_cache_dir(run_id: str) -> Path:
    """Return the workspace-local cache directory for one audit run."""

    return config.ensure_data_dir() / "cache" / "audit" / run_id


def codex_skill_path() -> Path:
    """Return the packaged Tonepath Codex skill path."""

    return package_resource_path("resources", "codex", "skills", "tonepath-dj", "SKILL.md")


def codex_audit_schema_path() -> Path:
    """Return the packaged Codex audit output schema."""

    return package_resource_path(
        "resources",
        "codex",
        "skills",
        "tonepath-dj",
        "schemas",
        "audit-output.schema.json",
    )


def request_intent_payload(plan: SessionPlan) -> dict[str, object]:
    """Return the normalized intent fields used by benchmark scenarios."""

    return {
        "source_state": plan.request.source_state,
        "target_state": plan.request.target_state,
        "duration_min": plan.request.duration_sec // 60,
        "constraints": request_constraints(plan.request),
    }


def package_resource_path(*parts: str) -> Path:
    """Return a filesystem path for a packaged Tonepath resource."""

    resource = resources.files("tonepath").joinpath(*parts)
    return Path(str(resource))


def codex_prompt(evidence_path: str, web: bool) -> str:
    """Return the prompt passed to Codex for one audit run."""

    web_instruction = "Use web search for source-backed context." if web else "Do not use web search."
    return "\n".join(
        [
            "<task>",
            "Audit a Tonepath local listening path using the packaged Tonepath Codex skill.",
            f"Skill path: {codex_skill_path()}",
            f"Evidence pack path: {evidence_path}",
            "Read the skill's field semantics, threshold guide, and examples before deciding.",
            web_instruction,
            "</task>",
            "<output_contract>",
            "Return JSON matching the provided output schema only.",
            "Every decision must be keep, demote, or reject.",
            "Every factual claim must cite local evidence or web evidence.",
            "</output_contract>",
            "<safety>",
            "Do not play audio, modify files, modify SQLite, or run Tonepath playback.",
            "Do not invent BPM, vocalness, genre, mood, lyrics, or instrumentation.",
            "</safety>",
        ]
    )


def load_codex_audit_result(path: Path) -> dict[str, object]:
    """Load and minimally validate one Codex audit JSON result."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError("Codex did not write an audit result file.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Codex audit result was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Codex audit result must be a JSON object.")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise RuntimeError("Codex audit result must contain a decisions list.")
    for item in decisions:
        validate_codex_decision(item)
    return payload


def codex_failure_message(exc: OSError | subprocess.CalledProcessError) -> str:
    """Return a compact Codex failure message without dumping noisy logs."""

    if isinstance(exc, subprocess.CalledProcessError):
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        if lines:
            return f"Codex audit failed: {lines[-1]}"
    return "Codex audit failed."


def validate_codex_decision(item: object) -> None:
    """Validate one Codex audit decision enough for safe rendering."""

    if not isinstance(item, dict):
        raise RuntimeError("Each Codex audit decision must be an object.")
    if item.get("decision") not in {"keep", "demote", "reject"}:
        raise RuntimeError("Codex audit decision must be keep, demote, or reject.")
    if not isinstance(item.get("track_id"), int):
        raise RuntimeError("Codex audit decision must include an integer track_id.")
    if not isinstance(item.get("reason"), str):
        raise RuntimeError("Codex audit decision must include a reason.")
    evidence = item.get("evidence_used")
    if not isinstance(evidence, list):
        raise RuntimeError("Codex audit decision must include evidence_used.")
    for entry in evidence:
        if not isinstance(entry, dict):
            raise RuntimeError("Codex evidence entries must be objects.")
        if entry.get("type") == "web" and not entry.get("url"):
            raise RuntimeError("Web evidence entries must include a URL.")
