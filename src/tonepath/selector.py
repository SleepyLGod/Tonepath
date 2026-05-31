"""Deterministic track selection for state paths."""

from __future__ import annotations

import json

from tonepath.affect import affect_phase_fit, affect_profile_from_enrichment
from tonepath.analysis import (
    AUDIO_SEPARATOR_FEATURE_SOURCE,
    DEMUCS_FEATURE_SOURCE,
    ESSENTIA_VOICE_FEATURE_SOURCE,
    loudness_to_unit,
)
from tonepath.db import TonepathStore
from tonepath.display import canonical_track_key
from tonepath.models import CandidateScore, ProfileRule, SessionPlan, SessionPhase, Track, TrackFeatures


QUIET_GENRES = ("ambient", "classical", "instrumental", "lofi", "lo-fi", "downtempo")
VOCAL_HEAVY_GENRES = ("pop", "rap", "hip-hop", "r&b")
LOW_STIM_PHASES = {"focus", "decompress", "soften", "settle", "calm", "hold"}
STRICT_LOW_STIM_PHASES = {"soften", "settle", "calm", "hold"}


def select_path(
    store: TonepathStore,
    plan: SessionPlan,
    limit_per_phase: int = 2,
    excluded_track_ids: set[int] | None = None,
    profile_enabled: bool = True,
) -> list[CandidateScore]:
    """Select tracks for every phase in a session plan."""

    tracks = store.list_tracks(effective_metadata=True)
    profile_rules = store.list_profile_rules() if profile_enabled else []
    selected: list[CandidateScore] = []
    used_ids: set[int] = set(excluded_track_ids or set())
    used_keys = {canonical_track_key(track) for track in tracks if track.id in used_ids}
    for phase in plan.phases:
        candidates = [
            score_track(store, track, phase, profile_rules=profile_rules)
            for track in tracks
            if track.id is not None and track.id not in used_ids
            and canonical_track_key(track) not in used_keys
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        added = 0
        for candidate in candidates:
            if added >= limit_per_phase:
                break
            key = canonical_track_key(candidate.track)
            if key in used_keys:
                continue
            selected.append(candidate)
            used_keys.add(key)
            added += 1
            if candidate.track.id is not None:
                used_ids.add(candidate.track.id)
    return selected


def score_track(
    store: TonepathStore,
    track: Track,
    phase: SessionPhase,
    profile_rules: list[ProfileRule] | None = None,
    profile_enabled: bool = True,
) -> CandidateScore:
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
            if is_low_stimulation_phase(phase):
                reasons.append("low-stimulation safety penalty adjusted the score")
        if phase.target_energy <= 0.45 and features.vocalness is not None and features.vocalness >= 0.75:
            score -= vocal_heavy_low_stim_penalty(phase)
            reasons.append("vocal-heavy track is risky for low-stimulation phase")
    else:
        reasons.append("audio features unavailable; selection uses metadata and feedback")
        if track.duration is None:
            score -= 4.0
            reasons.append("low-evidence/unverified audio candidate")

    enrichment = store.list_enrichment(track.id)
    affect_profile = affect_profile_from_enrichment(enrichment)
    affect_delta, affect_reasons = affect_phase_fit(affect_profile, phase)
    if affect_delta:
        score += affect_delta
    reasons.extend(affect_reasons)

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
            elif features.vocalness >= 0.5 and is_low_stimulation_phase(phase):
                score -= 0.6 * source_weight
                reasons.append("vocalness feature is inconclusive for no-vocals constraint")
                reasons.append("inconclusive vocalness is risky for strict no-vocals constraint")
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

    if profile_enabled:
        active_rules = profile_rules if profile_rules is not None else store.list_profile_rules()
        profile_delta, profile_reasons = profile_rule_adjustment(active_rules, track, features, phase)
        score += profile_delta
        reasons.extend(profile_reasons)

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


def is_low_stimulation_phase(phase: SessionPhase) -> bool:
    """Return whether a phase should be conservative about stimulation."""

    return phase.label in LOW_STIM_PHASES or phase.target_energy <= 0.45


def vocal_heavy_low_stim_penalty(phase: SessionPhase) -> float:
    """Return a vocal-heavy penalty for low-stimulation phases."""

    if phase.label in STRICT_LOW_STIM_PHASES:
        return 3.0
    if phase.target_energy <= 0.35:
        return 1.6
    return 1.0


def stimulation_risk_count(features: TrackFeatures, phase: SessionPhase) -> int:
    """Return count of high-stimulation feature risks for the current phase."""

    count = 0
    low_stim = is_low_stimulation_phase(phase)
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
    strict_low_stim = phase.label in STRICT_LOW_STIM_PHASES or phase.target_energy <= 0.35
    if features.bpm is not None:
        if phase.target_energy <= 0.35 and features.bpm > 125.0:
            penalty += min((features.bpm - 125.0) / 30.0, 1.0) * 4.2
        elif phase.label == "lift" and phase.target_energy <= 0.45 and features.bpm > 130.0:
            penalty += min((features.bpm - 130.0) / 25.0, 1.0) * 3.5
        elif phase.target_energy <= 0.45 and features.bpm > 135.0:
            penalty += min((features.bpm - 135.0) / 35.0, 1.0) * 2.2
        elif phase.label in {"decompress", "soften", "settle", "calm"} and phase.target_energy <= 0.25 and features.bpm > 125.0:
            penalty += min((features.bpm - 125.0) / 30.0, 1.0) * 4.2
        elif phase.label == "focus" and features.bpm > 110.0:
            penalty += min((features.bpm - 110.0) / 35.0, 1.0) * 2.8
        elif phase.label in {"settle", "calm"} and features.bpm > 125.0:
            penalty += min((features.bpm - 125.0) / 30.0, 1.0) * 4.5
        elif phase.label in {"decompress", "soften"} and features.bpm > 130.0:
            penalty += min((features.bpm - 130.0) / 40.0, 1.0) * 1.6
        elif phase.label == "stabilize" and features.bpm > 130.0:
            penalty += min((features.bpm - 130.0) / 35.0, 1.0) * 3.0
        if strict_low_stim and features.bpm > 135.0:
            penalty += min((features.bpm - 135.0) / 20.0, 1.0) * 4.0
        if phase.label == "focus" and phase.target_energy <= 0.45 and features.bpm > 140.0:
            penalty += min((features.bpm - 140.0) / 10.0, 1.0) * 6.0

    if features.energy is not None:
        if phase.target_energy <= 0.35 and features.energy > phase.target_energy + 0.15:
            penalty += min((features.energy - phase.target_energy - 0.15) / 0.3, 1.0) * 2.0
        elif phase.target_energy <= 0.45 and features.energy > phase.target_energy + 0.2:
            penalty += min((features.energy - phase.target_energy - 0.2) / 0.3, 1.0) * 2.4
        elif phase.label == "focus" and features.energy > 0.58:
            penalty += min((features.energy - 0.58) / 0.28, 1.0) * 1.4
        elif phase.target_energy <= 0.4 and features.energy > phase.target_energy + 0.2:
            penalty += min((features.energy - phase.target_energy - 0.2) / 0.3, 1.0) * 1.1
        elif phase.label == "stabilize" and features.energy > 0.7:
            penalty += min((features.energy - 0.7) / 0.25, 1.0) * 0.8
        if strict_low_stim:
            soft_limit = 0.58 if phase.target_energy <= 0.25 else 0.62
            if features.energy > soft_limit:
                penalty += min((features.energy - soft_limit) / 0.16, 1.0) * 2.8
        elif phase.label == "focus" and phase.target_energy <= 0.45 and features.energy > 0.62:
            penalty += min((features.energy - 0.62) / 0.16, 1.0) * 1.6
        elif phase.label == "lift" and phase.target_energy <= 0.45 and features.energy > 0.68:
            penalty += min((features.energy - 0.68) / 0.18, 1.0) * 1.2

    if features.arousal_estimate is not None:
        if phase.target_arousal <= 0.35 and features.arousal_estimate > phase.target_arousal + 0.2:
            penalty += min((features.arousal_estimate - phase.target_arousal - 0.2) / 0.35, 1.0) * 2.2
        elif phase.target_arousal <= 0.45 and features.arousal_estimate > phase.target_arousal + 0.2:
            penalty += min((features.arousal_estimate - phase.target_arousal - 0.2) / 0.35, 1.0) * 1.6

    if features.loudness is not None:
        loudness_unit = loudness_to_unit(features.loudness)
        if phase.target_energy <= 0.35 and loudness_unit > phase.target_energy + 0.2:
            penalty += min((loudness_unit - phase.target_energy - 0.2) / 0.3, 1.0) * 1.4
        elif phase.target_energy <= 0.45 and loudness_unit > phase.target_energy + 0.25:
            penalty += min((loudness_unit - phase.target_energy - 0.25) / 0.3, 1.0) * 1.6
        elif phase.label == "focus" and loudness_unit > 0.6:
            penalty += min((loudness_unit - 0.6) / 0.25, 1.0) * 1.0
        elif phase.target_energy <= 0.4 and loudness_unit > phase.target_energy + 0.25:
            penalty += min((loudness_unit - phase.target_energy - 0.25) / 0.3, 1.0) * 0.9
        elif phase.label == "stabilize" and features.loudness >= -9.0:
            penalty += 0.8
        if strict_low_stim and features.loudness > -10.0:
            penalty += min((features.loudness + 10.0) / 4.0, 1.0) * 1.4
        elif phase.label == "focus" and phase.target_energy <= 0.45 and features.loudness > -9.5:
            penalty += min((features.loudness + 9.5) / 4.0, 1.0) * 1.0
        elif phase.label == "lift" and phase.target_energy <= 0.45 and features.loudness > -8.5:
            penalty += min((features.loudness + 8.5) / 4.0, 1.0) * 1.0
    if strict_low_stim and features.vocalness is not None and features.vocalness > 0.7:
        penalty += min((features.vocalness - 0.7) / 0.25, 1.0) * 0.8
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


def profile_rule_adjustment(
    rules: list[ProfileRule], track: Track, features: TrackFeatures | None, phase: SessionPhase
) -> tuple[float, list[str]]:
    """Return score adjustments from active profile rules."""

    score = 0.0
    reasons: list[str] = []
    for rule in rules:
        try:
            payload = json.loads(rule.value)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not profile_scope_matches(str(payload.get("scope", "global")), phase):
            continue
        rule_type = str(payload.get("rule_type", ""))
        weight = float_payload(payload.get("weight"), 0.5)
        threshold = float_payload(payload.get("threshold"), 0.0)
        delta = profile_rule_delta(rule_type, threshold, weight, str(payload.get("target", "")), track, features)
        if delta == 0.0:
            continue
        score += delta
        reasons.append(f"profile rule: {payload.get('rationale', rule_type)}")
    return score, reasons


def profile_scope_matches(scope: str, phase: SessionPhase) -> bool:
    """Return whether a profile rule applies to one phase."""

    return scope in {"global", phase.label}


def profile_rule_delta(rule_type: str, threshold: float, weight: float, target: str, track: Track, features: TrackFeatures | None) -> float:
    """Return one profile rule score delta."""

    if rule_type == "prefer_artist":
        return weight if track.artist and target and target.lower() in track.artist.lower() else 0.0
    if features is None:
        return 0.0
    if rule_type == "prefer_lower_loudness" and features.loudness is not None:
        return weight if features.loudness <= threshold else -weight
    if rule_type == "prefer_lower_energy" and features.energy is not None:
        return weight if features.energy <= threshold else -weight
    if rule_type == "prefer_lower_vocalness" and features.vocalness is not None:
        return weight if features.vocalness <= threshold else -weight
    if rule_type == "demote_high_bpm" and features.bpm is not None:
        return -weight if features.bpm >= threshold else 0.0
    return 0.0


def float_payload(value: object, default: float) -> float:
    """Return a float from a profile rule payload."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default
