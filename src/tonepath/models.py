"""Shared data models for Tonepath core behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FeedbackType = Literal["like", "skip", "too-loud", "too-slow", "no-vocals"]
EnrichmentTier = Literal["local", "features", "online"]


@dataclass(frozen=True)
class Track:
    """A local audio track known to Tonepath."""

    id: int | None
    path: Path
    file_hash: str
    mtime: float
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    duration: float | None
    format: str | None


@dataclass(frozen=True)
class TrackFeatures:
    """Optional audio features derived from local analysis."""

    track_id: int
    bpm: float | None = None
    loudness: float | None = None
    energy: float | None = None
    vocalness: float | None = None
    arousal_estimate: float | None = None
    valence_estimate: float | None = None
    feature_source: str = "metadata"
    confidence: str = "low"


@dataclass(frozen=True)
class SessionRequest:
    """A user's requested state transition."""

    prompt: str
    source_state: str
    target_state: str
    duration_sec: int
    no_vocals: bool = False
    quiet: bool = False


@dataclass(frozen=True)
class SessionPhase:
    """A phase in a state-transition listening path."""

    label: str
    start_sec: int
    end_sec: int
    target_arousal: float
    target_valence: float
    target_energy: float
    vocal_policy: str = "allow"


@dataclass(frozen=True)
class SessionPlan:
    """A deterministic path from a current state to a target state."""

    request: SessionRequest
    phases: tuple[SessionPhase, ...]


@dataclass(frozen=True)
class CandidateScore:
    """A selected track and the score components behind it."""

    track: Track
    phase: SessionPhase
    score: float
    confidence: str
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EnrichmentRecord:
    """A source-attributed enrichment field for one local track."""

    track_id: int
    field: str
    value: str
    tier: EnrichmentTier
    source: str
    confidence: str
    is_online: bool = False


@dataclass(frozen=True)
class ProfileRule:
    """A local, explainable user preference rule."""

    id: int | None
    key: str
    value: str
    source: str
    confidence: str
