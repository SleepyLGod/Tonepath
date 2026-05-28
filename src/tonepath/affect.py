"""Derived music affect profile helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from tonepath.models import EnrichmentRecord, SessionPhase


DERIVED_AFFECT_SOURCE = "derived-affect-profile-v1"
AFFECT_FIELDS = ("sadness", "uplift", "calmness", "tension", "warmth", "darkness", "brightness")

TAG_AXIS_WEIGHTS: dict[str, dict[str, float]] = {
    "sad": {"sadness": 1.0, "darkness": 0.4},
    "melancholic": {"sadness": 0.9, "darkness": 0.4},
    "emotional": {"sadness": 0.35, "warmth": 0.35},
    "dark": {"darkness": 1.0, "tension": 0.35},
    "heavy": {"darkness": 0.5, "tension": 0.6},
    "dramatic": {"tension": 0.55, "darkness": 0.35},
    "drama": {"tension": 0.45, "darkness": 0.25},
    "epic": {"tension": 0.4, "brightness": 0.35},
    "powerful": {"tension": 0.5, "uplift": 0.25},
    "energetic": {"uplift": 0.55, "brightness": 0.45, "tension": 0.25},
    "fast": {"uplift": 0.35, "brightness": 0.3, "tension": 0.35},
    "upbeat": {"uplift": 0.8, "brightness": 0.55},
    "uplifting": {"uplift": 1.0, "brightness": 0.45, "warmth": 0.35},
    "happy": {"uplift": 0.9, "brightness": 0.7},
    "positive": {"uplift": 0.85, "brightness": 0.45, "warmth": 0.25},
    "hopeful": {"uplift": 0.75, "warmth": 0.55, "brightness": 0.35},
    "inspiring": {"uplift": 0.75, "warmth": 0.45, "brightness": 0.35},
    "motivational": {"uplift": 0.75, "brightness": 0.4},
    "fun": {"uplift": 0.65, "brightness": 0.55},
    "funny": {"uplift": 0.55, "brightness": 0.5},
    "summer": {"uplift": 0.45, "brightness": 0.65},
    "calm": {"calmness": 1.0, "warmth": 0.3},
    "relaxing": {"calmness": 0.95, "warmth": 0.25},
    "meditative": {"calmness": 0.9, "warmth": 0.2},
    "soft": {"calmness": 0.65, "warmth": 0.45},
    "slow": {"calmness": 0.55},
    "background": {"calmness": 0.55},
    "soundscape": {"calmness": 0.55, "darkness": 0.15},
    "nature": {"calmness": 0.45, "warmth": 0.35},
    "dream": {"calmness": 0.35, "warmth": 0.35, "brightness": 0.15},
    "love": {"warmth": 0.75, "uplift": 0.25},
    "romantic": {"warmth": 0.7, "sadness": 0.15},
    "melodic": {"warmth": 0.45, "brightness": 0.25},
    "ballad": {"warmth": 0.35, "sadness": 0.25},
    "cool": {"calmness": 0.3, "brightness": 0.25},
}


def derive_affect_profile(
    tags: Mapping[str, float],
    *,
    arousal: float | None = None,
    valence: float | None = None,
) -> dict[str, float]:
    """Return a normalized, explainable affect-axis profile from model evidence."""

    totals = {axis: 0.0 for axis in AFFECT_FIELDS}
    weights = {axis: 0.0 for axis in AFFECT_FIELDS}
    for raw_label, raw_score in tags.items():
        label = normalize_tag(raw_label)
        score = clamp(raw_score)
        for axis, weight in TAG_AXIS_WEIGHTS.get(label, {}).items():
            totals[axis] += score * weight
            weights[axis] += weight

    profile = {axis: (totals[axis] / weights[axis] if weights[axis] else 0.0) for axis in AFFECT_FIELDS}
    if valence is not None:
        positive = clamp(valence)
        profile["uplift"] = max(profile["uplift"], positive * 0.75)
        profile["brightness"] = max(profile["brightness"], positive * 0.65)
        profile["sadness"] = max(profile["sadness"], (1.0 - positive) * 0.55)
        profile["darkness"] = max(profile["darkness"], (1.0 - positive) * 0.45)
    if arousal is not None:
        active = clamp(arousal)
        profile["calmness"] = max(profile["calmness"], (1.0 - active) * 0.7)
        profile["tension"] = max(profile["tension"], active * 0.45)
    return {axis: round(clamp(value), 3) for axis, value in profile.items()}


def tags_to_scores(tags: object) -> dict[str, float]:
    """Return tag scores from a worker payload without trusting malformed rows."""

    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
        return {}
    scores: dict[str, float] = {}
    for item in tags:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            continue
        label, score = item
        try:
            scores[str(label)] = float(score)
        except (TypeError, ValueError):
            continue
    return scores


def affect_enrichment_records(track_id: int, profile: Mapping[str, float]) -> list[EnrichmentRecord]:
    """Build enrichment records for derived affect axes."""

    records: list[EnrichmentRecord] = []
    for axis in AFFECT_FIELDS:
        if axis not in profile:
            continue
        records.append(
            EnrichmentRecord(
                track_id=track_id,
                field=f"affect:{axis}",
                value=f"{clamp(profile[axis]):.3f}",
                tier="features",
                source=DERIVED_AFFECT_SOURCE,
                confidence="medium",
            )
        )
    return records


def affect_profile_from_enrichment(records: Iterable[EnrichmentRecord]) -> dict[str, float]:
    """Return the stored derived affect profile for one track."""

    profile: dict[str, float] = {}
    for record in records:
        if not record.field.startswith("affect:"):
            continue
        axis = record.field.removeprefix("affect:")
        if axis not in AFFECT_FIELDS:
            continue
        try:
            profile[axis] = clamp(float(record.value))
        except ValueError:
            continue
    return profile


def affect_phase_fit(profile: Mapping[str, float], phase: SessionPhase) -> tuple[float, list[str]]:
    """Return an affect-axis score adjustment and concise reasons for one phase."""

    if not profile:
        return 0.0, []
    score = 0.0
    reasons: list[str] = ["affect profile contributes to phase fit"]
    sadness = profile.get("sadness", 0.0)
    uplift = profile.get("uplift", 0.0)
    calmness = profile.get("calmness", 0.0)
    tension = profile.get("tension", 0.0)
    warmth = profile.get("warmth", 0.0)
    darkness = profile.get("darkness", 0.0)
    brightness = profile.get("brightness", 0.0)

    if phase.label == "hold":
        score += 0.45 * calmness + 0.35 * warmth
        score -= 0.35 * tension + 0.25 * darkness
    elif phase.label == "lift":
        score += 0.55 * uplift + 0.4 * warmth + 0.3 * brightness
        penalty = 0.65 * sadness + 0.55 * darkness + 0.45 * tension
        score -= penalty
        if penalty >= 0.45:
            reasons.append("sad/dark/tension affect is risky for the lift phase")
    elif phase.label in {"soften", "settle", "calm", "decompress", "focus", "stabilize"}:
        score += 0.35 * calmness + 0.25 * warmth
        score -= 0.4 * tension
        if phase.label in {"settle", "calm", "focus"}:
            score -= 0.25 * darkness
    else:
        score += 0.2 * uplift + 0.15 * brightness + 0.15 * warmth
    return score, reasons


def normalize_tag(label: str) -> str:
    """Normalize one model tag label for dictionary lookup."""

    return label.strip().lower().replace("mood/theme---", "").replace("mood/theme-", "").replace("_", " ")


def clamp(value: float) -> float:
    """Clamp one numeric affect value into the public 0..1 range."""

    return max(0.0, min(float(value), 1.0))
