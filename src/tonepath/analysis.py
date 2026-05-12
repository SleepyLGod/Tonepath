"""Local audio feature analysis for Tonepath."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import wave
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures


FEATURE_SOURCE = "basic-local-analysis"
FFMPEG_TIMEOUT_SEC = 30.0
FFMPEG_AUDIO_SUFFIXES = {".mp3", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wav"}
MEAN_VOLUME_PATTERN = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def analyze_library(store: TonepathStore, features: str = "basic") -> tuple[int, int]:
    """Analyze scanned local tracks and return analyzed/skipped counts."""

    if features != "basic":
        raise ValueError("Only basic feature analysis is implemented.")

    analyzed = 0
    skipped = 0
    for track in store.list_tracks():
        if track.id is None or not track.path.exists():
            skipped += 1
            continue
        result = analyze_track_basic(track)
        store.upsert_features(result)
        analyzed += 1
    return analyzed, skipped


def analyze_track_basic(track: Track) -> TrackFeatures:
    """Analyze one track with lightweight local-only feature extraction."""

    if track.id is None:
        raise ValueError("Track must be persisted before analysis.")

    if track.path.suffix.lower() == ".wav":
        wave_features = analyze_wave_file(track.path)
        if wave_features is not None:
            return TrackFeatures(
                track_id=track.id,
                loudness=wave_features[0],
                energy=wave_features[1],
                feature_source=FEATURE_SOURCE,
                confidence="medium",
            )

    ffmpeg_features = analyze_with_ffmpeg(track.path)
    if ffmpeg_features is not None:
        return TrackFeatures(
            track_id=track.id,
            loudness=ffmpeg_features[0],
            energy=ffmpeg_features[1],
            feature_source=FEATURE_SOURCE,
            confidence="medium",
        )

    return TrackFeatures(
        track_id=track.id,
        feature_source=FEATURE_SOURCE,
        confidence="low",
    )


def analyze_with_ffmpeg(path: Path) -> tuple[float, float] | None:
    """Return approximate loudness and energy for formats ffmpeg can decode."""

    if path.suffix.lower() not in FFMPEG_AUDIO_SUFFIXES or shutil.which("ffmpeg") is None:
        return None

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-vn",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_SEC,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    match = MEAN_VOLUME_PATTERN.search(result.stderr or "")
    if match is None:
        return None
    loudness = float(match.group(1))
    return loudness, loudness_to_unit(loudness)


def analyze_wave_file(path: Path) -> tuple[float, float] | None:
    """Return approximate loudness dBFS and normalized energy for a PCM WAV file."""

    try:
        with wave.open(str(path), "rb") as handle:
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError, EOFError):
        return None

    samples = pcm_samples(frames, sample_width)
    if not samples:
        return None

    max_amplitude = float(2 ** (sample_width * 8 - 1))
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms <= 0:
        loudness = -60.0
        energy = 0.0
    else:
        loudness = max(20.0 * math.log10(rms / max_amplitude), -60.0)
        energy = min(max(rms / max_amplitude, 0.0), 1.0)
    return loudness, energy


def pcm_samples(frames: bytes, sample_width: int) -> list[int]:
    """Decode little-endian PCM samples from WAV frame bytes."""

    if sample_width not in {1, 2, 3, 4}:
        return []

    samples: list[int] = []
    for index in range(0, len(frames) - sample_width + 1, sample_width):
        chunk = frames[index : index + sample_width]
        if sample_width == 1:
            samples.append(chunk[0] - 128)
        else:
            samples.append(int.from_bytes(chunk, byteorder="little", signed=True))
    return samples


def loudness_to_unit(loudness: float) -> float:
    """Map music-track mean dBFS loudness into a rough 0..1 intensity range."""

    return min(max((loudness + 30.0) / 30.0, 0.0), 1.0)
