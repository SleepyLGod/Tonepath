"""Read-only selection evaluation helpers."""

from __future__ import annotations

import math

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


def evaluate_suite(store: TonepathStore, limit: int, prompts: tuple[str, ...] = DEFAULT_EVAL_PROMPTS) -> list[dict[str, object]]:
    """Return product-oriented selection quality checks for multiple prompts."""

    payload: list[dict[str, object]] = []
    for prompt in prompts:
        plan = plan_session(prompt)
        rows = [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, limit)]
        annotate_red_flags(rows, no_vocals=plan.request.no_vocals)
        payload.append(
            {
                "prompt": prompt,
                "source_state": plan.request.source_state,
                "target_state": plan.request.target_state,
                "duration_min": plan.request.duration_sec // 60,
                "constraints": ["avoid_vocals"] if plan.request.no_vocals else [],
                "red_flag_count": sum(len(row["red_flags"]) for row in rows),
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


def float_or_none(value: object) -> float | None:
    """Return a float for numeric evaluation values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
