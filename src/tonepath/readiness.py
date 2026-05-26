"""Shared library readiness checks for CLI and TUI surfaces."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from tonepath import config as tonepath_config
from tonepath.db import TonepathStore
from tonepath.display import dirty_metadata_issues, display_label, duplicate_track_count
from tonepath.models import Track


@dataclass(frozen=True)
class LibraryStatus:
    """Current local library readiness counts."""

    tracks: int
    features: int
    missing_features: int
    vocalness: int
    mir: int
    tags: int
    dirty_metadata: int = 0
    duplicate_tracks: int = 0
    tracks_outside_music_dirs: int = 0
    suggested_music_dir: str | None = None
    missing_analysis_tracks: tuple[str, ...] = ()


def library_status(store: TonepathStore) -> LibraryStatus:
    """Return local library readiness counts."""

    tracks = store.list_tracks()
    outside_tracks = tracks_outside_configured_dirs(tracks)
    row = store.conn.execute(
        """
        SELECT
          COUNT(t.id) AS tracks,
          COUNT(f.track_id) AS features,
          SUM(CASE WHEN f.track_id IS NULL THEN 1 ELSE 0 END) AS missing_features,
          SUM(CASE WHEN f.vocalness IS NOT NULL THEN 1 ELSE 0 END) AS vocalness,
          SUM(CASE WHEN f.energy IS NOT NULL AND f.loudness IS NOT NULL AND f.bpm IS NOT NULL THEN 1 ELSE 0 END) AS mir,
          (
            SELECT COUNT(DISTINCT track_id)
            FROM track_enrichment
            WHERE field LIKE 'tag:%'
          ) AS tags
        FROM tracks t
        LEFT JOIN track_features f ON f.track_id = t.id
        """
    ).fetchone()
    if row is None:
        return LibraryStatus(0, 0, 0, 0, 0, 0)
    return LibraryStatus(
        tracks=int(row["tracks"] or 0),
        features=int(row["features"] or 0),
        missing_features=int(row["missing_features"] or 0),
        vocalness=int(row["vocalness"] or 0),
        mir=int(row["mir"] or 0),
        tags=int(row["tags"] or 0),
        dirty_metadata=sum(1 for track in tracks if dirty_metadata_issues(track)),
        duplicate_tracks=duplicate_track_count(tracks),
        tracks_outside_music_dirs=len(outside_tracks),
        suggested_music_dir=suggested_music_dir(outside_tracks),
        missing_analysis_tracks=missing_analysis_track_labels(store, limit=5),
    )


def missing_analysis_track_labels(store: TonepathStore, limit: int) -> tuple[str, ...]:
    """Return display labels for tracks missing feature rows or core MIR fields."""

    rows = store.conn.execute(
        """
        SELECT t.*
        FROM tracks t
        LEFT JOIN track_features f ON f.track_id = t.id
        WHERE f.track_id IS NULL
           OR f.energy IS NULL
           OR f.loudness IS NULL
           OR f.bpm IS NULL
        ORDER BY t.path
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    labels: list[str] = []
    for row in rows:
        track = Track(
            id=int(row["id"]),
            path=Path(row["path"]),
            file_hash=str(row["file_hash"]),
            mtime=float(row["mtime"]),
            title=row["title"],
            artist=row["artist"],
            album=row["album"],
            genre=row["genre"],
            duration=row["duration"],
            format=row["format"],
        )
        labels.append(f"{escape(display_label(track))} ({escape(display_relative_path(track.path))})")
    return tuple(labels)


def tracks_outside_configured_dirs(tracks: list[Track]) -> list[Track]:
    """Return tracks not covered by configured music directories."""

    roots = [path.expanduser().resolve() for path in tonepath_config.load_config().expanded_music_dirs()]
    outside: list[Track] = []
    for track in tracks:
        resolved_path = track.path.expanduser().resolve()
        if not any(resolved_path.is_relative_to(root) for root in roots):
            outside.append(track)
    return outside


def suggested_music_dir(tracks: list[Track]) -> str | None:
    """Return a likely directory to add to config for uncovered tracks."""

    if not tracks:
        return None
    parents = [str(track.path.expanduser().resolve().parent) for track in tracks]
    try:
        common = Path(os.path.commonpath(parents))
    except ValueError:
        common = Path(parents[0])
    try:
        relative = common.relative_to(Path.cwd())
    except ValueError:
        return str(common)
    if str(relative) != ".":
        return str(relative)
    return str(common)


def display_relative_path(path: Path) -> str:
    """Return a path relative to the current directory when possible."""

    try:
        return str(path.expanduser().resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def status_next_action(status: LibraryStatus, runtime_ready: bool, settings: tonepath_config.TonepathConfig) -> str:
    """Return one concise next action for normal users."""

    if status.tracks == 0:
        return "Add a music directory, then run `uv run tonepath prepare`."
    if status.tracks_outside_music_dirs:
        if status.suggested_music_dir:
            return f"Run `uv run tonepath config add-music-dir {status.suggested_music_dir}`, then `uv run tonepath prepare`."
        return "Add the active library directory to config, then run `uv run tonepath prepare`."
    if status.missing_features or status.mir < status.tracks:
        if status.features > 0 or status.mir > 0:
            return "Review or replace files with missing analysis, then run `uv run tonepath prepare`."
        return "Run `uv run tonepath prepare`."
    if settings.models.mode == "fast":
        return "Ready for TUI. Run `uv run tonepath`."
    if status.vocalness < status.tracks or status.tags < status.tracks:
        if runtime_ready:
            return "Run `uv run tonepath prepare` for model-backed tags."
        if settings.models.mode == "full":
            return "Run `uv run tonepath models setup essentia-tf`, then `uv run tonepath prepare --full`."
        return "Ready for TUI; run `uv run tonepath models setup essentia-tf` for better vocalness."
    if status.duplicate_tracks or status.dirty_metadata:
        return "Ready for TUI; review duplicate candidates or dirty metadata when recommendations look odd."
    return "Ready for TUI. Run `uv run tonepath`."


def readiness_label(status: LibraryStatus, runtime_ready: bool, settings: tonepath_config.TonepathConfig) -> str:
    """Return a compact readiness state for normal users."""

    if status.tracks == 0 or status.tracks_outside_music_dirs:
        return "Needs setup"
    if status.missing_features or status.mir < status.tracks:
        if status.features > 0 or status.mir > 0:
            return "Review files"
        return "Needs preparation"
    if settings.models.mode != "fast" and (status.vocalness < status.tracks or status.tags < status.tracks):
        return "Needs preparation" if runtime_ready else "Model setup available"
    return "Ready for TUI"


def quality_check_hint(status: LibraryStatus) -> str:
    """Return a concise benchmark hint for the current library state."""

    if status.tracks == 0:
        return "Prepare a library before running benchmark checks."
    if status.missing_features or status.mir < status.tracks:
        return "Resolve missing analysis before relying on `uv run tonepath eval suite --limit 8`."
    return "Run `uv run tonepath eval suite --limit 8` after prepare."


def readiness_blocks_session(label: str) -> bool:
    """Return whether normal TUI flow should refuse a new path."""

    return label in {"Needs setup", "Needs preparation", "Review files"}
