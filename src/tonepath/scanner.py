"""Local music library scanner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from tonepath.models import Track


SUPPORTED_EXTENSIONS = {".mp3", ".flac", ".m4a", ".mp4", ".wav", ".aiff", ".aif", ".ogg"}


def scan_directory(root: Path) -> list[Track]:
    """Scan a local directory for supported audio files."""

    root = root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Music directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Music path is not a directory: {root}")

    tracks: list[Track] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            tracks.append(read_track(path))
    return tracks


def read_track(path: Path) -> Track:
    """Read metadata from one local audio file."""

    stat = path.stat()
    tags = read_tags(path)
    return Track(
        id=None,
        path=path.expanduser().resolve(),
        file_hash=fingerprint(path),
        mtime=stat.st_mtime,
        title=tags.get("title") or path.stem,
        artist=tags.get("artist"),
        album=tags.get("album"),
        genre=tags.get("genre"),
        duration=tags.get("duration"),
        format=path.suffix.lower().lstrip("."),
    )


def read_tags(path: Path) -> dict[str, Any]:
    """Read tags with Mutagen when available, falling back to filename metadata."""

    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return {}

    try:
        audio = MutagenFile(path, easy=True)
    except Exception:
        return {}
    if audio is None:
        return {}

    def first(key: str) -> str | None:
        values = audio.tags.get(key) if audio.tags else None
        if not values:
            return None
        return str(values[0])

    duration = None
    if getattr(audio, "info", None) is not None and getattr(audio.info, "length", None) is not None:
        duration = float(audio.info.length)

    return {
        "title": first("title"),
        "artist": first("artist"),
        "album": first("album"),
        "genre": first("genre"),
        "duration": duration,
    }


def fingerprint(path: Path) -> str:
    """Return a stable lightweight content fingerprint for a local file."""

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(stat.st_size - 1024 * 1024, 0))
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()

