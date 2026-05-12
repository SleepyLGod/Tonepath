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
PCM_ANALYSIS_RATE = 11025
PCM_ANALYSIS_SECONDS = 90
VOCALNESS_SECONDS = 45
MEAN_VOLUME_PATTERN = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


def analyze_library(store: TonepathStore, features: str = "basic") -> tuple[int, int]:
    """Analyze scanned local tracks and return analyzed/skipped counts."""

    if features not in {"basic", "vocalness"}:
        raise ValueError("Only basic and vocalness feature analysis are implemented.")

    analyzed = 0
    skipped = 0
    for track in store.list_tracks():
        if track.id is None or not track.path.exists():
            skipped += 1
            continue
        existing = store.get_features(track.id)
        if features == "basic":
            result = analyze_track_basic(track, existing)
        else:
            result = analyze_track_vocalness(track, existing)
        store.upsert_features(result)
        analyzed += 1
    return analyzed, skipped


def analyze_track_basic(track: Track, existing: TrackFeatures | None = None) -> TrackFeatures:
    """Analyze one track with lightweight local-only feature extraction."""

    if track.id is None:
        raise ValueError("Track must be persisted before analysis.")

    if track.path.suffix.lower() == ".wav":
        wave_features = analyze_wave_file(track.path)
        if wave_features is not None:
            return TrackFeatures(
                track_id=track.id,
                bpm=wave_features[2],
                loudness=wave_features[0],
                energy=wave_features[1],
                vocalness=existing.vocalness if existing else None,
                arousal_estimate=existing.arousal_estimate if existing else None,
                valence_estimate=existing.valence_estimate if existing else None,
                feature_source=FEATURE_SOURCE,
                confidence="medium",
            )

    ffmpeg_features = analyze_with_ffmpeg(track.path)
    if ffmpeg_features is not None:
        return TrackFeatures(
            track_id=track.id,
            bpm=ffmpeg_features[2],
            loudness=ffmpeg_features[0],
            energy=ffmpeg_features[1],
            vocalness=existing.vocalness if existing else None,
            arousal_estimate=existing.arousal_estimate if existing else None,
            valence_estimate=existing.valence_estimate if existing else None,
            feature_source=FEATURE_SOURCE,
            confidence="medium",
        )

    return TrackFeatures(
        track_id=track.id,
        bpm=existing.bpm if existing else None,
        loudness=existing.loudness if existing else None,
        energy=existing.energy if existing else None,
        vocalness=existing.vocalness if existing else None,
        arousal_estimate=existing.arousal_estimate if existing else None,
        valence_estimate=existing.valence_estimate if existing else None,
        feature_source=FEATURE_SOURCE,
        confidence=existing.confidence if existing else "low",
    )


def analyze_track_vocalness(track: Track, existing: TrackFeatures | None = None) -> TrackFeatures:
    """Analyze one track for conservative local spectral vocalness."""

    if track.id is None:
        raise ValueError("Track must be persisted before analysis.")

    samples: list[int] | None = None
    sample_rate = PCM_ANALYSIS_RATE
    if track.path.suffix.lower() == ".wav":
        wave_pcm = decode_wave_pcm(track.path)
        if wave_pcm is not None:
            samples, sample_rate = wave_pcm
    if samples is None:
        samples = decode_pcm_with_ffmpeg(track.path, PCM_ANALYSIS_RATE, VOCALNESS_SECONDS)
        sample_rate = PCM_ANALYSIS_RATE

    vocalness = estimate_vocalness(samples, sample_rate) if samples is not None else None
    confidence = "medium" if vocalness is not None else (existing.confidence if existing else "low")
    return TrackFeatures(
        track_id=track.id,
        bpm=existing.bpm if existing else None,
        loudness=existing.loudness if existing else None,
        energy=existing.energy if existing else None,
        vocalness=vocalness,
        arousal_estimate=existing.arousal_estimate if existing else None,
        valence_estimate=existing.valence_estimate if existing else None,
        feature_source=FEATURE_SOURCE,
        confidence=confidence,
    )


def analyze_with_ffmpeg(path: Path) -> tuple[float, float, float | None] | None:
    """Return approximate loudness, energy, and optional BPM for ffmpeg-decodable formats."""

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
    return loudness, loudness_to_unit(loudness), analyze_bpm_with_ffmpeg(path)


def analyze_bpm_with_ffmpeg(path: Path) -> float | None:
    """Estimate BPM from a short local ffmpeg PCM decode."""

    samples = decode_pcm_with_ffmpeg(path, PCM_ANALYSIS_RATE, PCM_ANALYSIS_SECONDS)
    if samples is None:
        return None
    return estimate_bpm(samples, PCM_ANALYSIS_RATE)


def decode_pcm_with_ffmpeg(path: Path, sample_rate: int, seconds: int) -> list[int] | None:
    """Decode one audio file to mono signed 16-bit PCM samples with ffmpeg."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-t",
        str(seconds),
        "-f",
        "s16le",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    stdout = getattr(result, "stdout", b"")
    if result.returncode != 0 or not stdout:
        return None
    return pcm_samples(stdout, 2)


def analyze_wave_file(path: Path) -> tuple[float, float, float | None] | None:
    """Return approximate loudness dBFS, normalized energy, and optional BPM for a PCM WAV file."""

    try:
        with wave.open(str(path), "rb") as handle:
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
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
    return loudness, energy, estimate_bpm(samples, sample_rate)


def decode_wave_pcm(path: Path) -> tuple[list[int], int] | None:
    """Decode a PCM WAV file into samples and sample rate."""

    try:
        with wave.open(str(path), "rb") as handle:
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, OSError, EOFError):
        return None

    samples = pcm_samples(frames, sample_width)
    if not samples:
        return None
    return samples, sample_rate


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


def estimate_bpm(samples: list[int], sample_rate: int) -> float | None:
    """Estimate BPM from a coarse energy envelope, returning None when ambiguous."""

    if sample_rate <= 0 or len(samples) < sample_rate * 8:
        return None

    frame_size = max(sample_rate // 10, 1)
    envelope: list[float] = []
    for index in range(0, len(samples) - frame_size + 1, frame_size):
        frame = samples[index : index + frame_size]
        envelope.append(sum(abs(sample) for sample in frame) / len(frame))
    if len(envelope) < 80:
        return None

    mean = sum(envelope) / len(envelope)
    centered = [max(value - mean, 0.0) for value in envelope]
    if max(centered, default=0.0) <= 0.0:
        return None

    onset = [max(centered[index] - centered[index - 1], 0.0) for index in range(1, len(centered))]
    if sum(1 for value in onset if value > 0) < 8:
        return None

    frame_rate = sample_rate / frame_size
    min_lag = max(int(frame_rate * 60.0 / 180.0), 1)
    max_lag = max(int(frame_rate * 60.0 / 50.0), min_lag + 1)
    best_lag = 0
    best_score = 0.0
    total_energy = sum(value * value for value in onset)
    if total_energy <= 0.0:
        return None

    for lag in range(min_lag, min(max_lag + 1, len(onset) // 2)):
        score = sum(onset[index] * onset[index - lag] for index in range(lag, len(onset))) / total_energy
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag <= 0 or best_score < 0.25:
        return None
    bpm = 60.0 * frame_rate / best_lag
    if bpm < 50.0 or bpm > 180.0:
        return None
    return round(bpm, 1)


def estimate_vocalness(samples: list[int], sample_rate: int) -> float | None:
    """Estimate vocalness from simple spectral concentration features."""

    if sample_rate <= 0 or len(samples) < sample_rate * 8:
        return None

    window_size = min(max(sample_rate, 1024), len(samples))
    windows: list[list[int]] = []
    for index in range(0, len(samples) - window_size + 1, window_size):
        window = samples[index : index + window_size]
        rms = math.sqrt(sum(sample * sample for sample in window) / len(window))
        if rms > 500:
            windows.append(window)
    if len(windows) < 6:
        return None

    scores: list[float] = []
    for window in windows[:VOCALNESS_SECONDS]:
        low = band_power(window, sample_rate, (80.0, 100.0, 120.0, 180.0, 240.0, 320.0))
        mid = band_power(window, sample_rate, (500.0, 900.0, 1400.0, 2200.0))
        high = band_power(window, sample_rate, (3200.0, 4500.0))
        total = low + mid + high
        if total <= 0.0:
            continue
        mid_ratio = mid / total
        high_ratio = high / total
        zcr = zero_crossing_rate(window)
        vocal_score = 0.15 + 0.85 * mid_ratio
        if 0.02 <= zcr <= 0.18:
            vocal_score += 0.15
        if high_ratio > 0.42:
            vocal_score -= 0.2
        scores.append(min(max(vocal_score, 0.0), 1.0))

    if len(scores) < 4:
        return None
    average = sum(scores) / len(scores)
    if 0.42 <= average <= 0.58:
        return None
    return round(average, 2)


def band_power(samples: list[int], sample_rate: int, frequencies: tuple[float, ...]) -> float:
    """Return average Goertzel power for a few representative frequencies."""

    if not samples:
        return 0.0
    return sum(goertzel_power(samples, sample_rate, frequency) for frequency in frequencies) / len(frequencies)


def goertzel_power(samples: list[int], sample_rate: int, frequency: float) -> float:
    """Return normalized power around one target frequency."""

    if sample_rate <= 0 or frequency <= 0.0:
        return 0.0
    coefficient = 2.0 * math.cos(2.0 * math.pi * frequency / sample_rate)
    previous = 0.0
    previous2 = 0.0
    for sample in samples:
        value = float(sample) + coefficient * previous - previous2
        previous2 = previous
        previous = value
    power = previous2 * previous2 + previous * previous - coefficient * previous * previous2
    return max(power / (len(samples) * len(samples)), 0.0)


def zero_crossing_rate(samples: list[int]) -> float:
    """Return the fraction of adjacent samples that change sign."""

    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (previous < 0 <= sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample
    return crossings / (len(samples) - 1)


def loudness_to_unit(loudness: float) -> float:
    """Map music-track mean dBFS loudness into a rough 0..1 intensity range."""

    return min(max((loudness + 30.0) / 30.0, 0.0), 1.0)
