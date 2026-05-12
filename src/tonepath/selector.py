"""Deterministic track selection for state paths."""

from __future__ import annotations

from tonepath.analysis import loudness_to_unit
from tonepath.db import TonepathStore
from tonepath.models import CandidateScore, SessionPlan, SessionPhase, Track, TrackFeatures


QUIET_GENRES = ("ambient", "classical", "instrumental", "lofi", "lo-fi", "downtempo")
VOCAL_HEAVY_GENRES = ("pop", "rap", "hip-hop", "r&b")


def select_path(
    store: TonepathStore,
    plan: SessionPlan,
    limit_per_phase: int = 2,
    excluded_track_ids: set[int] | None = None,
) -> list[CandidateScore]:
    """Select tracks for every phase in a session plan."""

    tracks = store.list_tracks()
    selected: list[CandidateScore] = []
    used_ids: set[int] = set(excluded_track_ids or set())
    for phase in plan.phases:
        candidates = [
            score_track(store, track, phase)
            for track in tracks
            if track.id is not None and track.id not in used_ids
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        for candidate in candidates[:limit_per_phase]:
            selected.append(candidate)
            if candidate.track.id is not None:
                used_ids.add(candidate.track.id)
    return selected


def score_track(store: TonepathStore, track: Track, phase: SessionPhase) -> CandidateScore:
    """Score one track against one phase using explainable components."""

    if track.id is None:
        raise ValueError("Track must be persisted before scoring.")

    features = store.get_features(track.id)
    feedback = store.feedback_counts_for_track(track.id)
    score = 0.0
    reasons: list[str] = []
    confidence = "low"

    genre = (track.genre or "").lower()
    if phase.target_energy <= 0.4 and any(token in genre for token in QUIET_GENRES):
        score += 2.0
        reasons.append("genre fits a lower-energy phase")
    elif phase.target_energy >= 0.6 and not any(token in genre for token in QUIET_GENRES):
        score += 0.5
        reasons.append("genre may fit a higher-energy phase")

    if features:
        confidence = features.confidence
        score += feature_fit(features, phase)
        reasons.append(f"audio features available from {features.feature_source}")
        if features.energy is not None:
            reasons.append("energy feature contributes to phase fit")
        if features.loudness is not None:
            reasons.append("loudness feature contributes to phase fit")
    else:
        reasons.append("audio features unavailable; selection uses metadata and feedback")

    if phase.vocal_policy == "avoid":
        if features and features.vocalness is not None:
            if features.vocalness <= 0.35:
                score += 2.0
                reasons.append("vocalness feature supports no-vocals constraint")
            else:
                score -= 3.0
                reasons.append("vocalness feature conflicts with no-vocals constraint")
        elif any(token in genre for token in VOCAL_HEAVY_GENRES):
            score -= 1.0
            reasons.append("genre may be vocal-heavy; confidence is low")
        else:
            reasons.append("no-vocals requested but vocalness is unknown")

    score += feedback.get("like", 0) * 1.5
    score -= feedback.get("skip", 0) * 2.0
    score -= feedback.get("too-loud", 0) * 1.0
    score -= feedback.get("too-slow", 0) * 0.5
    score -= feedback.get("no-vocals", 0) * 1.0

    if feedback:
        reasons.append("previous local feedback adjusted the score")

    if track.duration:
        score += 0.2
        reasons.append("duration is known")

    return CandidateScore(track=track, phase=phase, score=score, confidence=confidence, reasons=tuple(reasons))


def feature_fit(features: TrackFeatures, phase: SessionPhase) -> float:
    """Return a simple distance-based feature fit score."""

    score = 0.0
    if features.energy is not None:
        score += 1.0 - abs(features.energy - phase.target_energy)
    if features.loudness is not None:
        score += 1.0 - abs(loudness_to_unit(features.loudness) - phase.target_energy)
    if features.arousal_estimate is not None:
        score += 1.0 - abs(features.arousal_estimate - phase.target_arousal)
    if features.valence_estimate is not None:
        score += 1.0 - abs(features.valence_estimate - phase.target_valence)
    return score
