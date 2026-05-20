"""Deterministic track selection for state paths."""

from __future__ import annotations

from tonepath.analysis import (
    AUDIO_SEPARATOR_FEATURE_SOURCE,
    DEMUCS_FEATURE_SOURCE,
    ESSENTIA_VOICE_FEATURE_SOURCE,
    loudness_to_unit,
)
from tonepath.db import TonepathStore
from tonepath.models import CandidateScore, SessionPlan, SessionPhase, Track, TrackFeatures


QUIET_GENRES = ("ambient", "classical", "instrumental", "lofi", "lo-fi", "downtempo")
VOCAL_HEAVY_GENRES = ("pop", "rap", "hip-hop", "r&b")
LOW_STIM_PHASES = {"focus", "decompress", "soften", "settle", "calm"}


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
        stimulation_penalty = phase_stimulation_penalty(features, phase)
        score -= stimulation_penalty
        reasons.append(f"audio features available from {features.feature_source}")
        if features.energy is not None:
            reasons.append("energy feature contributes to phase fit")
        if features.loudness is not None:
            reasons.append("loudness feature contributes to phase fit")
        if features.bpm is not None:
            reasons.append("BPM feature contributes to phase fit")
        if stimulation_penalty:
            reasons.append("phase stimulation penalty adjusted the score")
    else:
        reasons.append("audio features unavailable; selection uses metadata and feedback")

    if phase.vocal_policy == "avoid":
        if features and features.vocalness is not None:
            source_weight = vocalness_source_weight(features.feature_source)
            if features.vocalness <= 0.35:
                bonus = 2.0 * source_weight
                offset = no_vocals_stimulation_offset(features, phase, bonus)
                score += bonus - offset
                reasons.append("vocalness feature supports no-vocals constraint")
                if offset:
                    reasons.append("low vocalness but overstimulating for this phase")
            elif features.vocalness <= 0.4:
                bonus = 1.0 * source_weight
                offset = no_vocals_stimulation_offset(features, phase, bonus)
                score += bonus - offset
                reasons.append("vocalness feature weakly supports no-vocals constraint")
                if offset:
                    reasons.append("low vocalness but overstimulating for this phase")
            elif features.vocalness >= 0.65:
                score -= 3.0 * source_weight
                reasons.append("vocalness feature conflicts with no-vocals constraint")
            else:
                reasons.append("vocalness feature is inconclusive for no-vocals constraint")
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
    if features.bpm is not None:
        score += bpm_fit(features.bpm, phase)
    return score


def vocalness_source_weight(feature_source: str) -> float:
    """Return how strongly selector should trust one vocalness source."""

    if feature_source == ESSENTIA_VOICE_FEATURE_SOURCE:
        return 1.4
    if feature_source in {AUDIO_SEPARATOR_FEATURE_SOURCE, DEMUCS_FEATURE_SOURCE}:
        return 0.8
    return 1.0


def no_vocals_stimulation_offset(features: TrackFeatures, phase: SessionPhase, bonus: float) -> float:
    """Return how much low-vocalness bonus should be offset by stimulation risk."""

    risk_count = stimulation_risk_count(features, phase)
    if risk_count == 0:
        return 0.0
    return min(2.0 * risk_count, bonus * 0.95)


def stimulation_risk_count(features: TrackFeatures, phase: SessionPhase) -> int:
    """Return count of high-stimulation feature risks for the current phase."""

    count = 0
    low_stim = phase.label in LOW_STIM_PHASES
    if features.bpm is not None:
        if low_stim and features.bpm >= 140.0:
            count += 1
        elif phase.label == "stabilize" and features.bpm >= 140.0:
            count += 1
    if features.energy is not None:
        if low_stim and features.energy >= 0.68:
            count += 1
        elif phase.label == "stabilize" and features.energy >= 0.72:
            count += 1
    if features.loudness is not None:
        if (low_stim or phase.label == "stabilize") and features.loudness >= -9.0:
            count += 1
    return count


def phase_stimulation_penalty(features: TrackFeatures, phase: SessionPhase) -> float:
    """Return penalties for tracks that are too stimulating for a phase."""

    penalty = 0.0
    if features.bpm is not None:
        if phase.label in {"decompress", "soften", "settle", "calm"} and phase.target_energy <= 0.25 and features.bpm > 125.0:
            penalty += min((features.bpm - 125.0) / 30.0, 1.0) * 4.2
        elif phase.label == "focus" and features.bpm > 110.0:
            penalty += min((features.bpm - 110.0) / 35.0, 1.0) * 2.8
        elif phase.label in {"settle", "calm"} and features.bpm > 125.0:
            penalty += min((features.bpm - 125.0) / 30.0, 1.0) * 4.5
        elif phase.label in {"decompress", "soften"} and features.bpm > 130.0:
            penalty += min((features.bpm - 130.0) / 40.0, 1.0) * 1.6
        elif phase.label == "stabilize" and features.bpm > 130.0:
            penalty += min((features.bpm - 130.0) / 35.0, 1.0) * 3.0

    if features.energy is not None:
        if phase.label == "focus" and features.energy > 0.58:
            penalty += min((features.energy - 0.58) / 0.28, 1.0) * 1.4
        elif phase.target_energy <= 0.4 and features.energy > phase.target_energy + 0.2:
            penalty += min((features.energy - phase.target_energy - 0.2) / 0.3, 1.0) * 1.1
        elif phase.label == "stabilize" and features.energy > 0.7:
            penalty += min((features.energy - 0.7) / 0.25, 1.0) * 0.8

    if features.loudness is not None:
        loudness_unit = loudness_to_unit(features.loudness)
        if phase.label == "focus" and loudness_unit > 0.6:
            penalty += min((loudness_unit - 0.6) / 0.25, 1.0) * 1.0
        elif phase.target_energy <= 0.4 and loudness_unit > phase.target_energy + 0.25:
            penalty += min((loudness_unit - phase.target_energy - 0.25) / 0.3, 1.0) * 0.9
        elif phase.label == "stabilize" and features.loudness >= -9.0:
            penalty += 0.8
    return penalty


def bpm_fit(bpm: float, phase: SessionPhase) -> float:
    """Return a conservative BPM fit score for a phase target."""

    target_bpm = 70.0 + phase.target_energy * 90.0
    score = 0.75 * (1.0 - min(abs(bpm - target_bpm) / 80.0, 1.0))
    if phase.target_energy <= 0.5 and bpm > 140.0:
        score -= 0.5
    if phase.target_energy >= 0.6 and 90.0 <= bpm <= 165.0:
        score += 0.25
    return score
