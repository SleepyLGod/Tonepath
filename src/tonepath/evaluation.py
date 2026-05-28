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
from tonepath.affect import affect_profile_from_enrichment
from tonepath.benchmark import aggregate_result, evaluate_benchmark_scenario, load_benchmark_scenarios, scenario_from_prompt
from tonepath.db import TonepathStore
from tonepath.display import (
    canonical_track_key,
    dirty_metadata_issues_from_values,
    display_artist,
    display_label,
    display_title,
    normalize_key_part,
)
from tonepath.embedding import cosine_similarity, read_clap_audio_embedding, read_or_create_clap_text_embedding, read_or_create_clap_text_embeddings
from tonepath.models import CandidateScore, ProfileRule, SessionPlan, TrackFeatures
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


PROFILE_RISK_SCAN_LIMIT = 20
PROFILE_COMPANION_WARNING = "Lower-vocalness rule may need a high-BPM demotion companion."
PROFILE_COVERAGE_WARNING = "Profile risk may be hidden by --limit; rerun with --limit 20 to check high-BPM companion risk."


def evaluate_profile_comparison(store: TonepathStore, prompt: str, limit: int) -> dict[str, object]:
    """Compare selection with and without active profile rules."""

    no_profile = evaluate_selection(store, prompt, limit, profile_enabled=False)
    with_profile = evaluate_selection(store, prompt, limit, profile_enabled=True)
    no_candidates = no_profile.get("candidates", [])
    with_candidates = with_profile.get("candidates", [])
    if not isinstance(no_candidates, list) or not isinstance(with_candidates, list):
        raise RuntimeError("Profile comparison requires candidate lists.")
    movements = profile_movements(no_candidates, with_candidates)
    active_rules = store.list_profile_rules()
    active_rule_count = len(active_rules)
    warnings = profile_comparison_warnings(active_rules, with_candidates)
    if not warnings and limit < PROFILE_RISK_SCAN_LIMIT:
        risk_scan = evaluate_selection(store, prompt, PROFILE_RISK_SCAN_LIMIT, profile_enabled=True)
        risk_candidates = risk_scan.get("candidates", [])
        if isinstance(risk_candidates, list) and profile_comparison_warnings(active_rules, risk_candidates):
            warnings = [PROFILE_COVERAGE_WARNING]
    return {
        "prompt": prompt,
        "active_rule_count": active_rule_count,
        "message": profile_comparison_message(active_rule_count, movements),
        "warnings": warnings,
        "no_profile": no_profile,
        "with_profile": with_profile,
        "movements": movements,
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


def profile_movements(no_candidates: list[object], with_candidates: list[object]) -> list[dict[str, object]]:
    """Return rank and score movement for with-profile candidates."""

    no_index: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for index, row in enumerate(no_candidates, start=1):
        if not isinstance(row, dict):
            continue
        identity = candidate_identity(row)
        if not identity:
            continue
        no_index.setdefault(identity, []).append((index, row))
    movements: list[dict[str, object]] = []
    for with_rank, row in enumerate(with_candidates, start=1):
        if not isinstance(row, dict):
            continue
        identity = candidate_identity(row)
        matches = no_index.get(identity) if identity else None
        no_match = matches.pop(0) if matches else None
        no_rank = None if no_match is None else no_match[0]
        no_row = None if no_match is None else no_match[1]
        with_score = numeric_score(row)
        no_score = None if no_row is None else numeric_score(no_row)
        movements.append(
            {
                "track": row.get("track"),
                "phase": row.get("phase"),
                "rank_no_profile": no_rank,
                "rank_with_profile": with_rank,
                "rank_delta": None if no_rank is None else no_rank - with_rank,
                "score_no_profile": no_score,
                "score_with_profile": with_score,
                "score_delta": None if no_score is None or with_score is None else round(with_score - no_score, 3),
                "profile_reasons": profile_reasons(row),
            }
        )
    return movements


def candidate_identity(row: dict[str, object]) -> str:
    """Return a stable identity for one evaluation row."""

    track = row.get("track")
    if not isinstance(track, dict):
        return ""
    track_id = track.get("id")
    if track_id is not None:
        return f"id:{track_id}"
    canonical = track.get("canonical_key")
    if isinstance(canonical, list):
        key = "|".join(str(part) for part in canonical)
        return f"key:{key}" if key else ""
    label = track.get("display_label")
    return str(label) if label else ""


def numeric_score(row: dict[str, object]) -> float | None:
    """Return a row score as a float when available."""

    value = row.get("score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def profile_reasons(row: dict[str, object]) -> list[str]:
    """Return profile-specific candidate reasons."""

    reasons = row.get("reasons")
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons if "profile rule:" in str(reason)]


def profile_comparison_message(active_rule_count: int, movements: list[dict[str, object]]) -> str:
    """Return a concise profile comparison message."""

    if active_rule_count == 0:
        return "No active profile rules; recommendations are unchanged."
    if any(row.get("profile_reasons") for row in movements):
        return "Active profile rules changed selection scores."
    return "Active profile rules did not match the compared candidates; try a larger --limit or a prompt whose phases match the rule scope."


def profile_comparison_warnings(rules: list[ProfileRule], candidates: list[object]) -> list[str]:
    """Return warnings for risky profile-rule combinations."""

    parsed_rules = [payload for payload in (profile_rule_payload(rule) for rule in rules) if payload is not None]
    lower_vocal_scopes = {str(rule.get("scope") or "global") for rule in parsed_rules if rule.get("rule_type") == "prefer_lower_vocalness"}
    if not lower_vocal_scopes:
        return []
    demote_bpm_scopes = {str(rule.get("scope") or "global") for rule in parsed_rules if rule.get("rule_type") == "demote_high_bpm"}
    if any(high_bpm_without_companion(candidate, lower_vocal_scopes, demote_bpm_scopes) for candidate in candidates):
        return [PROFILE_COMPANION_WARNING]
    return []


def profile_rule_payload(rule: ProfileRule) -> dict[str, object] | None:
    """Return one profile rule JSON payload, or None if malformed."""

    try:
        payload = json.loads(rule.value)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def high_bpm_without_companion(candidate: object, lower_vocal_scopes: set[str], demote_bpm_scopes: set[str]) -> bool:
    """Return whether one candidate has high-BPM risk without a matching companion rule."""

    if not isinstance(candidate, dict):
        return False
    phase = str(candidate.get("phase") or "")
    lower_vocal_applies = "global" in lower_vocal_scopes or phase in lower_vocal_scopes
    demote_bpm_applies = "global" in demote_bpm_scopes or phase in demote_bpm_scopes
    if not lower_vocal_applies or demote_bpm_applies:
        return False
    features = candidate.get("features")
    bpm = float_or_none(features.get("bpm")) if isinstance(features, dict) else None
    if bpm is not None and bpm >= 135.0:
        return True
    reasons = candidate.get("reasons")
    return isinstance(reasons, list) and any("overstimulating" in str(reason) for reason in reasons)


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


BAKEOFF_ENGINES = {"selector", "clap"}
BAKEOFF_RESULT_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def evaluate_bakeoff(store: TonepathStore, engines: tuple[str, ...], limit: int) -> list[dict[str, object]]:
    """Compare selector output against optional experimental engines."""

    unsupported = [engine for engine in engines if engine not in BAKEOFF_ENGINES]
    if unsupported:
        raise RuntimeError(f"Unsupported bake-off engine(s): {', '.join(unsupported)}")
    scenarios = load_benchmark_scenarios()
    if "clap" in engines:
        read_or_create_clap_text_embeddings(clap_probes_for_scenarios(scenarios))
    payload: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_results = [evaluate_bakeoff_engine(store, scenario, engine, limit) for engine in engines]
        payload.append(
            {
                "scenario_id": scenario["id"],
                "lang": scenario["lang"],
                "prompt": scenario["prompt"],
                "limit": min(limit, int(scenario["limit"])),
                "engines": scenario_results,
                "delta": bakeoff_delta(scenario_results),
            }
        )
    return payload


def evaluate_bakeoff_engine(store: TonepathStore, scenario: dict[str, object], engine: str, limit: int) -> dict[str, object]:
    """Return one engine's benchmark result for a scenario."""

    prompt = str(scenario["prompt"])
    plan = plan_session(prompt)
    scenario_limit = min(limit, int(scenario["limit"]))
    if engine == "selector":
        candidates = eval_candidates(store, plan, scenario_limit)
    elif engine == "clap":
        candidates = clap_candidates(store, plan, scenario_limit)
    else:
        raise RuntimeError(f"Unsupported bake-off engine: {engine}")
    rows = [candidate_to_eval_row(store, candidate) for candidate in candidates]
    annotate_red_flags(rows, no_vocals=plan.request.no_vocals)
    annotate_yellow_flags(rows, no_vocals=plan.request.no_vocals)
    benchmark = evaluate_benchmark_scenario(scenario, request_intent_payload(plan), rows)
    if engine == "clap" and not rows:
        checks = list(benchmark["checks"])
        checks.append(
            {
                "type": "engine_candidates",
                "status": "fail",
                "message": "CLAP has no cached audio embeddings. Run analyze --features embedding --method clap.",
                "affected_ranks": [],
            }
        )
        benchmark = {**benchmark, "checks": checks, "result": aggregate_result(checks)}
    return {
        "engine": engine,
        "result": benchmark["result"],
        "checks": benchmark["checks"],
        "red_flag_count": sum(len(row["red_flags"]) for row in rows),
        "yellow_flag_count": sum(len(row["yellow_flags"]) for row in rows),
        "candidates": rows,
    }


def clap_candidates(store: TonepathStore, plan: SessionPlan, limit: int) -> list[CandidateScore]:
    """Return a CLAP text-audio reranked path from cached audio embeddings."""

    per_phase = max(1, math.ceil(limit / max(len(plan.phases), 1)))
    tracks = store.list_tracks()
    selected: list[CandidateScore] = []
    used_ids: set[int] = set()
    used_keys: set[tuple[str, str, int]] = set()
    for phase in plan.phases:
        probe = clap_probe_for_phase(plan, phase.label)
        text_embedding = read_or_create_clap_text_embedding(probe)
        scored: list[CandidateScore] = []
        for track in tracks:
            if track.id is None or track.id in used_ids or canonical_track_key(track) in used_keys:
                continue
            audio_embedding = read_clap_audio_embedding(track)
            if audio_embedding is None:
                continue
            score = cosine_similarity(audio_embedding, text_embedding)
            scored.append(
                CandidateScore(
                    track=track,
                    phase=phase,
                    score=score,
                    confidence="medium",
                    reasons=(f"CLAP text-audio similarity for probe: {probe}",),
                )
            )
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        for candidate in scored[:per_phase]:
            selected.append(candidate)
            key = canonical_track_key(candidate.track)
            used_keys.add(key)
            if candidate.track.id is not None:
                used_ids.add(candidate.track.id)
    return selected[:limit]


def clap_probes_for_scenarios(scenarios: list[dict[str, object]]) -> list[str]:
    """Return all deterministic CLAP probes needed by a benchmark run."""

    probes: list[str] = []
    for scenario in scenarios:
        plan = plan_session(str(scenario["prompt"]))
        probes.extend(clap_probe_for_phase(plan, phase.label) for phase in plan.phases)
    return probes


def clap_probe_for_phase(plan: SessionPlan, phase_label: str) -> str:
    """Return a deterministic English CLAP probe for one parsed intent and phase."""

    request = plan.request
    constraints = set(request_constraints(request))
    parts: list[str] = []
    if request.target_state == "uplift":
        parts.append("gentle uplifting warm calm music, not loud, not gloomy")
    elif request.target_state == "calm":
        if "gentle_uplift" in constraints:
            parts.append("calm reassuring warm music, not dark, not tense")
        else:
            parts.append("calm relaxing low stimulation music")
    elif request.target_state == "focus":
        parts.append("low distraction focus music")
    elif request.target_state == "energized":
        parts.append("gradually energizing music")
    elif request.target_state == "steady":
        parts.append("steady rhythmic music, not too loud")
    else:
        parts.append("balanced background music")

    if phase_label == "hold":
        parts.append("comforting, gentle, low arousal")
    elif phase_label == "stabilize":
        parts.append("stable, warm, controlled energy")
    elif phase_label == "lift":
        parts.append("hopeful, brighter, gently uplifting")
    elif phase_label in {"soften", "settle", "calm"}:
        parts.append("soothing, low tension, not gloomy")
    elif phase_label in {"decompress", "focus"}:
        parts.append("quiet, clear, low stimulation")

    if request.no_vocals:
        parts.append("instrumental, no vocals")
    if request.quiet:
        parts.append("low stimulation, not loud")
    return "; ".join(parts)


def bakeoff_delta(results: list[dict[str, object]]) -> dict[str, object]:
    """Return a compact comparison between selector and the first non-selector engine."""

    baseline = next((row for row in results if row.get("engine") == "selector"), None)
    compared = next((row for row in results if row.get("engine") != "selector"), None)
    if baseline is None or compared is None:
        return {"verdict": "inconclusive"}
    base_score = bakeoff_result_score(baseline)
    compared_score = bakeoff_result_score(compared)
    if compared_score < base_score:
        verdict = "improved"
    elif compared_score > base_score:
        verdict = "regressed"
    else:
        verdict = "inconclusive"
    return {
        "baseline_engine": baseline["engine"],
        "baseline_result": baseline["result"],
        "compared_engine": compared["engine"],
        "compared_result": compared["result"],
        "verdict": verdict,
    }


def bakeoff_result_score(result: dict[str, object]) -> tuple[int, int, int]:
    """Return an ordered score where lower means a better benchmark outcome."""

    checks = result.get("checks", [])
    warn_count = 0
    fail_count = 0
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status", "pass")).lower()
            if status == "warn":
                warn_count += 1
            elif status == "fail":
                fail_count += 1
    return (BAKEOFF_RESULT_ORDER.get(str(result.get("result")), 1), fail_count, warn_count)


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
    enrichment = store.list_enrichment(candidate.track.id) if candidate.track.id is not None else []
    affect_profile = affect_profile_from_enrichment(enrichment)
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
        "features": features_to_eval_row(features, affect_profile),
        "reasons": list(candidate.reasons),
        "red_flags": [],
        "yellow_flags": [],
    }


def features_to_eval_row(features: TrackFeatures | None, affect_profile: dict[str, float] | None = None) -> dict[str, object]:
    """Convert stored feature values into JSON-safe evaluation fields."""

    affect = affect_profile or {}
    if features is None:
        return {
            "source": None,
            "confidence": None,
            "energy": None,
            "loudness": None,
            "bpm": None,
            "vocalness": None,
            "arousal": None,
            "valence": None,
            "affect_profile": affect,
        }
    return {
        "source": features.feature_source,
        "confidence": features.confidence,
        "energy": round(features.energy, 3) if features.energy is not None else None,
        "loudness": round(features.loudness, 2) if features.loudness is not None else None,
        "bpm": round(features.bpm, 1) if features.bpm is not None else None,
        "vocalness": round(features.vocalness, 3) if features.vocalness is not None else None,
        "arousal": round(features.arousal_estimate, 3) if features.arousal_estimate is not None else None,
        "valence": round(features.valence_estimate, 3) if features.valence_estimate is not None else None,
        "affect_profile": affect,
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
    affect_profile = features.get("affect_profile")
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
    if phase == "lift" and isinstance(affect_profile, dict):
        if affect_float(affect_profile, "sadness") >= 0.6:
            flags.append("high sadness in lift candidate")
        if affect_float(affect_profile, "darkness") >= 0.6:
            flags.append("high darkness in lift candidate")
        if affect_float(affect_profile, "tension") >= 0.6:
            flags.append("high tension in lift candidate")
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
    affect_profile = features.get("affect_profile")
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
    if phase in {"hold", "stabilize", "lift"}:
        if not isinstance(affect_profile, dict) or not affect_profile:
            flags.append("affect evidence unknown")
        elif phase == "lift" and affect_float(affect_profile, "uplift") < 0.25:
            flags.append("low uplift evidence for lift phase")
    return flags


def float_or_none(value: object) -> float | None:
    """Return a float for numeric evaluation values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def affect_float(profile: dict[object, object], axis: str) -> float:
    """Return a numeric affect-axis value from a candidate profile."""

    value = profile.get(axis)
    number = float_or_none(value)
    return 0.0 if number is None else number


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
