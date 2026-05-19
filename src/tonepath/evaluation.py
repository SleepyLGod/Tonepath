"""Read-only selection evaluation helpers."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import CandidateScore, SessionPlan, TrackFeatures
from tonepath.planner import plan_session
from tonepath.selector import select_path


DEFAULT_EVAL_PROMPTS: tuple[str, ...] = (
    "我现在很烦，想半小时后进入写代码状态，不要人声",
    "我现在很累，想用二十分钟提神",
    "晚上想放松下来，三十分钟，低刺激",
    "深度工作四十五分钟，低刺激，不要人声",
)


def evaluate_selection(store: TonepathStore, prompt: str, limit: int) -> list[dict[str, object]]:
    """Return stable selection-evaluation rows without writing profile state."""

    plan = plan_session(prompt)
    return [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, limit)]


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
        "constraints": ["avoid_vocals"] if plan.request.no_vocals else [],
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
        subprocess.run(command, input=codex_prompt(evidence_path, web=web), text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Codex audit failed.") from exc
    result = load_codex_audit_result(result_path)
    return {
        "evidence": evidence,
        "codex": result,
        "codex_result_path": str(result_path),
        "web_enabled": web,
    }


def evaluate_suite(store: TonepathStore, limit: int, prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS) -> list[dict[str, object]]:
    """Return product-oriented selection quality checks for multiple prompts."""

    payload: list[dict[str, object]] = []
    for prompt in prompts:
        plan = plan_session(prompt)
        rows = [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, limit)]
        annotate_red_flags(rows, no_vocals=plan.request.no_vocals)
        annotate_yellow_flags(rows, no_vocals=plan.request.no_vocals)
        payload.append(
            {
                "prompt": prompt,
                "source_state": plan.request.source_state,
                "target_state": plan.request.target_state,
                "duration_min": plan.request.duration_sec // 60,
                "constraints": ["avoid_vocals"] if plan.request.no_vocals else [],
                "red_flag_count": sum(len(row["red_flags"]) for row in rows),
                "yellow_flag_count": sum(len(row["yellow_flags"]) for row in rows),
                "candidates": rows,
            }
        )
    return payload


def eval_candidates(store: TonepathStore, plan: SessionPlan, limit: int) -> list[CandidateScore]:
    """Return a small balanced candidate set for evaluation output."""

    per_phase = max(1, math.ceil(limit / max(len(plan.phases), 1)))
    return select_path(store, plan, limit_per_phase=per_phase)[:limit]


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


def annotate_red_flags(rows: list[dict[str, object]], no_vocals: bool) -> None:
    """Attach product-quality red flags to evaluation rows in place."""

    for index, row in enumerate(rows):
        row["red_flags"] = candidate_red_flags(row, rank=index + 1, no_vocals=no_vocals)


def annotate_yellow_flags(rows: list[dict[str, object]], no_vocals: bool) -> None:
    """Attach review warnings to evaluation rows in place."""

    for row in rows:
        row["yellow_flags"] = candidate_yellow_flags(row, no_vocals=no_vocals)


def candidate_red_flags(row: dict[str, object], rank: int, no_vocals: bool) -> list[str]:
    """Return product-quality red flags for one candidate."""

    if rank > 3:
        return []
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
        flags.append("high vocalness in no-vocals top 3")
    if confidence == "low" or features.get("source") is None:
        flags.append("low evidence in top 3")
    if phase in {"decompress", "focus"}:
        if energy is not None and energy >= 0.75:
            flags.append("high energy in calm/focus top 3")
        if loudness is not None and loudness >= -8.0:
            flags.append("high loudness in calm/focus top 3")
        if bpm is not None and bpm >= 150.0:
            flags.append("high BPM in calm/focus top 3")
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
    """Return the repo-local Tonepath Codex skill path."""

    return config.repo_root() / "codex" / "skills" / "tonepath-dj" / "SKILL.md"


def codex_audit_schema_path() -> Path:
    """Return the repo-local Codex audit output schema."""

    return config.repo_root() / "codex" / "skills" / "tonepath-dj" / "schemas" / "audit-output.schema.json"


def codex_prompt(evidence_path: str, web: bool) -> str:
    """Return the prompt passed to Codex for one audit run."""

    web_instruction = "Use web search for source-backed context." if web else "Do not use web search."
    return "\n".join(
        [
            "<task>",
            "Audit a Tonepath local listening path using the repo-local Codex skill.",
            f"Skill path: {codex_skill_path()}",
            f"Evidence pack path: {evidence_path}",
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
