"""Library hygiene reporting and local metadata overrides."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.display import (
    METADATA_ARTIST_FIELD,
    METADATA_TITLE_FIELD,
    canonical_track_key,
    clean_metadata_text,
    dirty_metadata_issues,
    display_artist,
    display_label,
    display_title,
)
from tonepath.models import Track


def library_issues(store: TonepathStore) -> dict[str, object]:
    """Return dirty metadata and duplicate candidate groups for a library."""

    raw_tracks = store.list_tracks()
    effective_tracks = {track.id: track for track in store.list_tracks(effective_metadata=True)}
    overrides = store.metadata_overrides_by_track()
    dirty_rows = [
        track_issue_row(raw_track, effective_tracks.get(raw_track.id, raw_track), overrides.get(raw_track.id or -1, {}))
        for raw_track in raw_tracks
        if dirty_metadata_issues(effective_tracks.get(raw_track.id, raw_track))
    ]
    duplicate_groups = [
        [track_issue_row(raw_track, effective_tracks.get(raw_track.id, raw_track), overrides.get(raw_track.id or -1, {})) for raw_track in group]
        for group in duplicate_track_groups(raw_tracks, effective_tracks)
    ]
    return {
        "summary": {
            "dirty_metadata": len(dirty_rows),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_candidates": sum(max(len(group) - 1, 0) for group in duplicate_groups),
        },
        "dirty_metadata": dirty_rows,
        "duplicate_groups": duplicate_groups,
    }


def duplicate_track_groups(raw_tracks: list[Track], effective_tracks: dict[int | None, Track]) -> list[list[Track]]:
    """Return raw tracks grouped by effective canonical display key."""

    grouped: dict[tuple[str, str, int], list[Track]] = defaultdict(list)
    for raw_track in raw_tracks:
        effective = effective_tracks.get(raw_track.id, raw_track)
        grouped[canonical_track_key(effective)].append(raw_track)
    return [group for group in grouped.values() if len(group) > 1]


def track_issue_row(raw_track: Track, effective_track: Track, overrides: dict[str, str]) -> dict[str, object]:
    """Return one stable library issue row."""

    return {
        "track_id": raw_track.id,
        "display_label": display_label(effective_track),
        "raw_title": raw_track.title,
        "raw_artist": raw_track.artist,
        "effective_title": display_title(effective_track),
        "effective_artist": display_artist(effective_track),
        "relative_path": display_relative_path(raw_track.path),
        "metadata_issues": dirty_metadata_issues(effective_track),
        "has_override": bool(overrides),
    }


def set_metadata_override(store: TonepathStore, track_id: int, title: str | None, artist: str | None) -> Track:
    """Apply local display metadata overrides and return the effective track."""

    if title is None and artist is None:
        raise ValueError("At least one of --title or --artist is required.")
    if store.get_track(track_id) is None:
        raise ValueError(f"Unknown track id: {track_id}")
    if title is not None:
        cleaned_title = clean_metadata_text(title)
        if cleaned_title is None:
            raise ValueError("--title must not be empty after cleaning.")
        store.upsert_metadata_override(track_id, METADATA_TITLE_FIELD, cleaned_title)
    if artist is not None:
        cleaned_artist = clean_metadata_text(artist)
        if cleaned_artist is None:
            raise ValueError("--artist must not be empty after cleaning.")
        store.upsert_metadata_override(track_id, METADATA_ARTIST_FIELD, cleaned_artist)
    track = store.get_track(track_id, effective_metadata=True)
    if track is None:
        raise RuntimeError("Metadata override was written but the track disappeared.")
    return track


def clear_metadata_override(store: TonepathStore, track_id: int) -> tuple[int, Track]:
    """Clear local display metadata overrides and return deleted count plus effective track."""

    if store.get_track(track_id) is None:
        raise ValueError(f"Unknown track id: {track_id}")
    deleted = store.clear_metadata_overrides(track_id)
    track = store.get_track(track_id, effective_metadata=True)
    if track is None:
        raise RuntimeError("Metadata override was cleared but the track disappeared.")
    return deleted, track


def display_relative_path(path: Path) -> str:
    """Return a path relative to the current directory when possible."""

    try:
        return str(path.expanduser().resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)
