"""Display and library hygiene helpers for local track metadata."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from tonepath.models import Track


NULL_MARKER = "(null)"
UNKNOWN = "unknown"


def clean_metadata_text(value: str | None) -> str | None:
    """Return display-safe metadata text without mutating stored values."""

    if value is None:
        return None
    cleaned = value.replace(NULL_MARKER, "").strip()
    return cleaned or None


def display_title(track: Track) -> str:
    """Return the best title for a track display."""

    return clean_metadata_text(track.title) or track.path.stem or UNKNOWN


def display_artist(track: Track) -> str:
    """Return the best artist for a track display."""

    return clean_metadata_text(track.artist) or UNKNOWN


def display_label(track: Track) -> str:
    """Return a compact title-artist label for a track."""

    return f"{display_title(track)} - {display_artist(track)}"


def fallback_track_label(title: str | None, fallback: str) -> str:
    """Return a display-safe title with a filename fallback."""

    return clean_metadata_text(title) or Path(fallback).stem or fallback


def canonical_track_key(track: Track) -> tuple[str, str, int]:
    """Return a stable duplicate-detection key for one local track."""

    duration_bucket = -1
    if track.duration is not None:
        duration_bucket = int(track.duration // 10.0)
    return (
        normalize_key_part(display_title(track)),
        normalize_key_part(display_artist(track)),
        duration_bucket,
    )


def normalize_key_part(value: str) -> str:
    """Normalize one display value for duplicate grouping."""

    lowered = value.casefold()
    return re.sub(r"[\W_]+", "", lowered)


def dirty_metadata_issues(track: Track) -> list[str]:
    """Return hygiene issues found in one track's raw metadata."""

    return dirty_metadata_issues_from_values(track.title, track.artist)


def dirty_metadata_issues_from_values(title: object, artist: object) -> list[str]:
    """Return hygiene issues for raw title and artist values."""

    issues: list[str] = []
    if dirty_title(title):
        issues.append("dirty title")
    if dirty_artist(artist):
        issues.append("dirty artist")
    return issues


def dirty_title(value: object) -> bool:
    """Return whether a raw title is missing or visibly polluted."""

    text = str(value).strip() if value is not None else ""
    return not text or NULL_MARKER in text


def dirty_artist(value: object) -> bool:
    """Return whether a raw artist is missing, unknown, or visibly polluted."""

    text = str(value).strip() if value is not None else ""
    return not text or NULL_MARKER in text or text.casefold() == UNKNOWN


def duplicate_track_count(tracks: list[Track]) -> int:
    """Return the number of tracks beyond the first in duplicate groups."""

    counts = Counter(canonical_track_key(track) for track in tracks)
    return sum(count - 1 for count in counts.values() if count > 1)
