"""Read-only selection evaluation helpers."""

from __future__ import annotations

import math

from tonepath.db import TonepathStore
from tonepath.models import CandidateScore, SessionPlan, TrackFeatures
from tonepath.planner import plan_session
from tonepath.selector import select_path


def evaluate_selection(store: TonepathStore, prompt: str, limit: int) -> list[dict[str, object]]:
    """Return stable selection-evaluation rows without writing profile state."""

    plan = plan_session(prompt)
    return [candidate_to_eval_row(store, candidate) for candidate in eval_candidates(store, plan, limit)]


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
