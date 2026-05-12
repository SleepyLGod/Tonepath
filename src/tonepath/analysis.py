"""Local audio feature analysis for Tonepath."""

from __future__ import annotations

import math
import wave
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures


FEATURE_SOURCE = "basic-local-analysis"


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

    return TrackFeatures(
        track_id=track.id,
        feature_source=FEATURE_SOURCE,
        confidence="low",
    )


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
    """Map dBFS loudness into a rough 0..1 intensity range."""

    return min(max((loudness + 60.0) / 60.0, 0.0), 1.0)
