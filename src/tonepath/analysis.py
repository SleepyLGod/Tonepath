"""Local audio feature analysis for Tonepath."""

from __future__ import annotations

import contextlib
import math
import os
import re
import shutil
import subprocess
import time
import wave
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.model_runtime import ensure_essentia_tf_runtime, run_essentia_tf_tags
from tonepath.models import EnrichmentRecord, Track, TrackFeatures
from tonepath.scanner import fingerprint, read_track


FEATURE_SOURCE = "basic-local-analysis"
ESSENTIA_MIR_FEATURE_SOURCE = "model-essentia-mir"
ESSENTIA_VOICE_FEATURE_SOURCE = "model-essentia-voice-instrumental"
ESSENTIA_TAGS_FEATURE_SOURCE = "model-essentia-tags"
ESSENTIA_TF_TAGS_FEATURE_SOURCE = "model-essentia-tf-tags"
DEMUCS_FEATURE_SOURCE = "model-demucs-cli"
AUDIO_SEPARATOR_FEATURE_SOURCE = "model-audio-separator"
FFMPEG_TIMEOUT_SEC = 30.0
DEMUCS_TIMEOUT_SEC = 900.0
SEPARATOR_TIMEOUT_SEC = 1200.0
FFMPEG_AUDIO_SUFFIXES = {".mp3", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".opus", ".wav"}
PCM_ANALYSIS_RATE = 11025
PCM_ANALYSIS_SECONDS = 90
VOCALNESS_SECONDS = 45
MEAN_VOLUME_PATTERN = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
VOCALNESS_METHODS = {"spectral", "demucs-cli", "audio-separator"}
TAGGING_METHODS = {"essentia", "essentia-tf"}
ANALYSIS_FEATURES = {"basic", "vocalness", "mir", "tags"}


@dataclass(frozen=True)
class AnalysisProgress:
    """A progress event from offline library analysis."""

    index: int
    total: int
    track: Track
    result: TrackFeatures | None = None
    runtime_sec: float | None = None
    error: str | None = None


def analyze_library(
    store: TonepathStore,
    features: str = "basic",
    method: str = "spectral",
    *,
    only_missing: bool = False,
    changed_only: bool = False,
    force: bool = False,
    limit: int | None = None,
    progress: Callable[[AnalysisProgress], None] | None = None,
) -> tuple[int, int]:
    """Analyze scanned local tracks and return analyzed/skipped counts."""

    if features not in ANALYSIS_FEATURES:
        raise ValueError("Only basic, vocalness, mir, and tags feature analysis are implemented.")
    if features == "vocalness" and method not in VOCALNESS_METHODS:
        raise ValueError("Only spectral, audio-separator, and demucs-cli vocalness methods are implemented.")
    if features == "mir" and method != "essentia":
        raise ValueError("Only essentia is supported for mir analysis.")
    if features == "tags" and method not in TAGGING_METHODS:
        raise ValueError("Only essentia and essentia-tf are supported for tags analysis.")
    if features == "basic" and method != "spectral":
        raise ValueError("The --method option is only supported with --features vocalness, mir, or tags.")
    if features == "vocalness" and method == "audio-separator" and shutil.which("audio-separator") is None:
        raise RuntimeError("audio-separator vocalness requires the optional models extra. Run: uv sync --extra models")
    if features == "vocalness" and method == "demucs-cli" and shutil.which("demucs") is None:
        raise RuntimeError("demucs-cli vocalness requires the `demucs` command on PATH. Tonepath does not install it.")
    if features == "mir" and method == "essentia":
        import_essentia_standard()
    if features == "tags" and method == "essentia":
        ensure_essentia_tagging_available()
    if features == "tags" and method == "essentia-tf":
        ensure_essentia_tf_runtime()
    if limit is not None and limit < 1:
        raise ValueError("Limit must be greater than zero.")
    if force and only_missing:
        raise ValueError("Use either force or only_missing, not both.")

    analyzed = 0
    skipped = 0
    selected, missing = select_tracks_for_analysis(
        store,
        features=features,
        method=method,
        only_missing=only_missing,
        changed_only=changed_only,
        force=force,
        limit=limit,
    )
    skipped += missing
    total = len(selected)
    for index, track in enumerate(selected, start=1):
        if track.id is None or not track.path.exists():
            skipped += 1
            continue
        existing = store.get_features(track.id)
        started = time.monotonic()
        result: TrackFeatures | None = None
        try:
            if features == "basic":
                result = analyze_track_basic(track, existing)
            elif features == "vocalness":
                result = analyze_track_vocalness(track, existing, method=method, force=force)
            elif features == "mir":
                result, enrichment = analyze_track_mir(track, existing, method=method)
                upsert_enrichment_records(store, enrichment)
            else:
                result, enrichment = analyze_track_tags(track, existing, method=method, force=force)
                upsert_enrichment_records(store, enrichment)
        except RuntimeError as exc:
            skipped += 1
            if progress is not None:
                progress(AnalysisProgress(index, total, track, None, time.monotonic() - started, str(exc)))
            continue
        if should_persist_result(existing, result, features, method):
            store.upsert_features(result)
        analyzed += 1
        if progress is not None:
            progress(AnalysisProgress(index, total, track, result, time.monotonic() - started))
    return analyzed, skipped


def select_tracks_for_analysis(
    store: TonepathStore,
    *,
    features: str,
    method: str,
    only_missing: bool,
    changed_only: bool,
    force: bool,
    limit: int | None,
) -> tuple[list[Track], int]:
    """Return tracks eligible for the requested local analysis pass."""

    selected: list[Track] = []
    skipped = 0
    for track in store.list_tracks():
        if track.id is None or not track.path.exists():
            skipped += 1
            continue
        existing = store.get_features(track.id)
        if not force and not track_needs_analysis(track, existing, features, method, only_missing, changed_only):
            skipped += 1
            continue
        selected.append(refresh_track_if_changed(store, track) if changed_only else track)
        if limit is not None and len(selected) >= limit:
            break
    return selected, skipped


def track_needs_analysis(
    track: Track,
    existing: TrackFeatures | None,
    features: str,
    method: str,
    only_missing: bool,
    changed_only: bool,
) -> bool:
    """Return whether one track should be analyzed in this pass."""

    if changed_only and not track_file_changed(track):
        return False
    if changed_only:
        return True
    if features == "basic":
        if only_missing:
            return existing is None or existing.energy is None or existing.loudness is None
        return True
    if features == "mir":
        if only_missing:
            return existing is None or existing.energy is None or existing.loudness is None or existing.bpm is None
        return True
    if features == "tags":
        if existing is not None and existing.feature_source == ESSENTIA_VOICE_FEATURE_SOURCE and existing.vocalness is not None:
            return False
        if only_missing:
            return existing is None or existing.vocalness is None or existing.feature_source != ESSENTIA_VOICE_FEATURE_SOURCE
        return True

    source = feature_source_for_method(method)
    if method in {"audio-separator", "demucs-cli"}:
        if existing is not None and existing.feature_source == ESSENTIA_VOICE_FEATURE_SOURCE:
            return False
        return existing is None or existing.vocalness is None or existing.feature_source != source
    if only_missing:
        return existing is None or existing.vocalness is None
    return True


def track_file_changed(track: Track) -> bool:
    """Return whether a file no longer matches its stored mtime or fingerprint."""

    try:
        stat = track.path.stat()
    except OSError:
        return True
    if abs(stat.st_mtime - track.mtime) > 0.0001:
        return True
    try:
        return fingerprint(track.path) != track.file_hash
    except OSError:
        return True


def refresh_track_if_changed(store: TonepathStore, track: Track) -> Track:
    """Refresh stale track metadata before analyzing changed files."""

    if not track_file_changed(track):
        return track
    refreshed = read_track(track.path)
    track_id = store.upsert_track(refreshed)
    return Track(
        id=track_id,
        path=refreshed.path,
        file_hash=refreshed.file_hash,
        mtime=refreshed.mtime,
        title=refreshed.title,
        artist=refreshed.artist,
        album=refreshed.album,
        genre=refreshed.genre,
        duration=refreshed.duration,
        format=refreshed.format,
    )


def feature_source_for_method(method: str) -> str:
    """Return the feature_source written by one vocalness method."""

    if method == "audio-separator":
        return AUDIO_SEPARATOR_FEATURE_SOURCE
    if method == "demucs-cli":
        return DEMUCS_FEATURE_SOURCE
    return FEATURE_SOURCE


def upsert_enrichment_records(store: TonepathStore, records: Sequence[EnrichmentRecord]) -> None:
    """Persist source-attributed enrichment records from one analysis pass."""

    for record in records:
        store.upsert_enrichment(record)


def feature_changed(existing: TrackFeatures | None, result: TrackFeatures) -> bool:
    """Return whether a new feature result should be persisted and counted."""

    if existing is None:
        return True
    return existing != result


def should_persist_result(existing: TrackFeatures | None, result: TrackFeatures, features: str, method: str) -> bool:
    """Return whether an analysis result should be stored."""

    if features == "vocalness" and method in {"audio-separator", "demucs-cli"}:
        if result.vocalness is None and existing is None:
            return False
    return feature_changed(existing, result)


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


def analyze_track_vocalness(
    track: Track,
    existing: TrackFeatures | None = None,
    method: str = "spectral",
    *,
    force: bool = False,
) -> TrackFeatures:
    """Analyze one track for local vocalness using the requested method."""

    if track.id is None:
        raise ValueError("Track must be persisted before analysis.")
    if method not in VOCALNESS_METHODS:
        raise ValueError("Only spectral, audio-separator, and demucs-cli vocalness methods are implemented.")
    if existing is not None and existing.feature_source == ESSENTIA_VOICE_FEATURE_SOURCE and method in {"audio-separator", "demucs-cli"} and not force:
        return existing
    if method == "audio-separator":
        vocalness = analyze_vocalness_with_audio_separator(track.path)
        if vocalness is None:
            return preserve_existing_vocalness(track.id, existing)
        return TrackFeatures(
            track_id=track.id,
            bpm=existing.bpm if existing else None,
            loudness=existing.loudness if existing else None,
            energy=existing.energy if existing else None,
            vocalness=vocalness,
            arousal_estimate=existing.arousal_estimate if existing else None,
            valence_estimate=existing.valence_estimate if existing else None,
            feature_source=AUDIO_SEPARATOR_FEATURE_SOURCE,
            confidence="high",
        )
    if method == "demucs-cli":
        vocalness = analyze_vocalness_with_demucs(track.path)
        if vocalness is None:
            return preserve_existing_vocalness(track.id, existing)
        return TrackFeatures(
            track_id=track.id,
            bpm=existing.bpm if existing else None,
            loudness=existing.loudness if existing else None,
            energy=existing.energy if existing else None,
            vocalness=vocalness,
            arousal_estimate=existing.arousal_estimate if existing else None,
            valence_estimate=existing.valence_estimate if existing else None,
            feature_source=DEMUCS_FEATURE_SOURCE,
            confidence="high",
        )

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


def analyze_track_mir(
    track: Track,
    existing: TrackFeatures | None = None,
    method: str = "essentia",
) -> tuple[TrackFeatures, list[EnrichmentRecord]]:
    """Analyze one track with an optional local MIR backend."""

    if track.id is None:
        raise ValueError("Track must be persisted before analysis.")
    if method != "essentia":
        raise ValueError("Only essentia MIR analysis is implemented.")

    values = extract_mir_with_essentia(track.path)
    features = TrackFeatures(
        track_id=track.id,
        bpm=number_or_none(values.get("bpm")),
        loudness=number_or_none(values.get("loudness")),
        energy=energy_from_mir_loudness(values.get("loudness")),
        vocalness=existing.vocalness if existing else None,
        arousal_estimate=existing.arousal_estimate if existing else None,
        valence_estimate=existing.valence_estimate if existing else None,
        feature_source=existing.feature_source if existing and existing.feature_source == ESSENTIA_VOICE_FEATURE_SOURCE else ESSENTIA_MIR_FEATURE_SOURCE,
        confidence=existing.confidence if existing and existing.feature_source == ESSENTIA_VOICE_FEATURE_SOURCE else "high",
    )
    enrichment = mir_enrichment_records(track.id, values)
    return features, enrichment


def extract_mir_with_essentia(path: Path) -> dict[str, object]:
    """Extract local MIR descriptors using Essentia MusicExtractor."""

    essentia_standard = import_essentia_standard()
    try:
        with suppress_native_output():
            results, _frames = essentia_standard.MusicExtractor()(str(path))
    except Exception as exc:  # pragma: no cover - Essentia raises C++ backed exceptions
        raise RuntimeError(f"Essentia MIR analysis failed for {path.name}.") from exc
    return {
        "bpm": descriptor_value(results, "rhythm.bpm"),
        "loudness": descriptor_value(results, "lowlevel.loudness_ebu128.integrated"),
        "danceability": descriptor_value(results, "rhythm.danceability"),
        "key": descriptor_value(results, "tonal.key_edma.key"),
        "scale": descriptor_value(results, "tonal.key_edma.scale"),
        "key_strength": descriptor_value(results, "tonal.key_edma.strength"),
        "dynamic_complexity": descriptor_value(results, "lowlevel.dynamic_complexity"),
    }


def import_essentia_standard() -> Any:
    """Import Essentia's standard API or raise a user-facing dependency error."""

    try:
        import essentia.standard as essentia_standard  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Essentia MIR analysis requires the optional extra. Run: uv sync --extra mir") from exc
    return essentia_standard


@contextlib.contextmanager
def suppress_native_output() -> Iterator[None]:
    """Suppress noisy native-library writes to stdout and stderr."""

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)


def descriptor_value(pool: Any, key: str) -> object | None:
    """Return one Essentia descriptor value when present."""

    try:
        return pool[key]
    except Exception:
        return None


def energy_from_mir_loudness(value: object) -> float | None:
    """Map a MIR loudness value into Tonepath's normalized energy slot."""

    loudness = number_or_none(value)
    return loudness_to_unit(loudness) if loudness is not None else None


def mir_enrichment_records(track_id: int, values: Mapping[str, object]) -> list[EnrichmentRecord]:
    """Build source-attributed enrichment records for MIR descriptors."""

    records: list[EnrichmentRecord] = []
    for field in ("key", "scale", "key_strength", "danceability", "dynamic_complexity"):
        value = values.get(field)
        if value is None:
            continue
        records.append(
            EnrichmentRecord(
                track_id=track_id,
                field=field,
                value=stringify_descriptor(value),
                tier="features",
                source=ESSENTIA_MIR_FEATURE_SOURCE,
                confidence="high",
            )
        )
    return records


def analyze_track_tags(
    track: Track,
    existing: TrackFeatures | None = None,
    method: str = "essentia",
    *,
    force: bool = False,
) -> tuple[TrackFeatures, list[EnrichmentRecord]]:
    """Analyze one track with optional local music-tagging models."""

    if track.id is None:
        raise ValueError("Track must be persisted before analysis.")
    if method not in TAGGING_METHODS:
        raise ValueError("Only essentia and essentia-tf tagging analysis is implemented.")
    if existing is not None and existing.feature_source == ESSENTIA_VOICE_FEATURE_SOURCE and existing.vocalness is not None and not force:
        return existing, []

    values = extract_tags_with_essentia_tf(track.path) if method == "essentia-tf" else extract_tags_with_essentia(track.path)
    vocalness = number_or_none(values.get("vocalness"))
    features = TrackFeatures(
        track_id=track.id,
        bpm=existing.bpm if existing else None,
        loudness=existing.loudness if existing else None,
        energy=existing.energy if existing else None,
        vocalness=vocalness if vocalness is not None else (existing.vocalness if existing else None),
        arousal_estimate=existing.arousal_estimate if existing else None,
        valence_estimate=existing.valence_estimate if existing else None,
        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE if vocalness is not None else (existing.feature_source if existing else FEATURE_SOURCE),
        confidence="high" if vocalness is not None else (existing.confidence if existing else "low"),
    )
    source = ESSENTIA_TF_TAGS_FEATURE_SOURCE if method == "essentia-tf" else ESSENTIA_TAGS_FEATURE_SOURCE
    return features, tag_enrichment_records(track.id, values, source=source)


def extract_tags_with_essentia(path: Path) -> dict[str, object]:
    """Extract local music tags with Essentia TensorFlow models when installed."""

    ensure_essentia_tagging_available()
    raise RuntimeError("Essentia tagging model files are not configured yet.")


def extract_tags_with_essentia_tf(path: Path) -> dict[str, object]:
    """Extract local music tags through the isolated Essentia TensorFlow runtime."""

    return run_essentia_tf_tags(path)


def ensure_essentia_tagging_available() -> None:
    """Raise a clear error when Essentia TensorFlow tagging support is missing."""

    essentia_standard = import_essentia_standard()
    if hasattr(essentia_standard, "TensorflowPredict2D"):
        return
    raise RuntimeError(
        "Essentia tagging requires TensorFlow model support, which is not available in this environment. "
        "MIR extraction is available with `uv sync --extra mir`; tagging models need a later optional adapter."
    )


def tag_enrichment_records(track_id: int, values: Mapping[str, object], source: str = ESSENTIA_TAGS_FEATURE_SOURCE) -> list[EnrichmentRecord]:
    """Build source-attributed enrichment records for music-tagging outputs."""

    records: list[EnrichmentRecord] = []
    tags = values.get("tags")
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
        return records
    for item in tags:
        if not isinstance(item, Sequence) or len(item) != 2:
            continue
        label, score = item
        records.append(
            EnrichmentRecord(
                track_id=track_id,
                field=f"tag:{label}",
                value=stringify_descriptor(score),
                tier="features",
                source=source,
                confidence="high",
            )
        )
    return records


def number_or_none(value: object) -> float | None:
    """Return a finite float value or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def stringify_descriptor(value: object) -> str:
    """Return a compact stable string for one enrichment descriptor."""

    number = number_or_none(value)
    if number is not None:
        return f"{number:.4g}"
    return str(value)


def analyze_vocalness_with_audio_separator(path: Path) -> float | None:
    """Run optional local audio-separator CLI separation and estimate vocalness."""

    if shutil.which("audio-separator") is None:
        raise RuntimeError("audio-separator vocalness requires the optional models extra. Run: uv sync --extra models")

    output_root = audio_separator_output_dir(path)
    model_root = audio_separator_model_dir()
    shutil.rmtree(output_root, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    command = [
        "audio-separator",
        "--single_stem",
        "Vocals",
        "--output_format",
        "WAV",
        "--output_dir",
        str(output_root),
        "--model_file_dir",
        str(model_root),
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=SEPARATOR_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    vocals_path = find_vocal_stem(output_root)
    if vocals_path is None:
        return None
    vocal_pcm = decode_wave_pcm(vocals_path)
    if vocal_pcm is None:
        return None
    vocal_samples, sample_rate = vocal_pcm
    mix_samples = decode_track_pcm(path, sample_rate, VOCALNESS_SECONDS)
    if mix_samples is None:
        return None
    return vocalness_from_stem_ratio(vocal_samples, mix_samples)


def audio_separator_model_dir() -> Path:
    """Return the local model cache directory for audio-separator."""

    return config.ensure_data_dir() / "cache" / "models" / "audio-separator"


def audio_separator_output_dir(path: Path) -> Path:
    """Return the local separated-output directory for one audio-separator run."""

    key = sha256(str(path.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
    return config.ensure_data_dir() / "cache" / "separated" / "audio-separator" / key


def preserve_existing_vocalness(track_id: int, existing: TrackFeatures | None = None) -> TrackFeatures:
    """Return a feature row that does not overwrite prior vocalness after model failure."""

    return TrackFeatures(
        track_id=track_id,
        bpm=existing.bpm if existing else None,
        loudness=existing.loudness if existing else None,
        energy=existing.energy if existing else None,
        vocalness=existing.vocalness if existing else None,
        arousal_estimate=existing.arousal_estimate if existing else None,
        valence_estimate=existing.valence_estimate if existing else None,
        feature_source=existing.feature_source if existing else FEATURE_SOURCE,
        confidence=existing.confidence if existing else "low",
    )


def analyze_vocalness_with_demucs(path: Path) -> float | None:
    """Run an optional local Demucs CLI separation and estimate vocalness from the vocal stem."""

    if shutil.which("demucs") is None:
        raise RuntimeError("demucs-cli vocalness requires the `demucs` command on PATH. Tonepath does not install it.")

    cache_root = demucs_track_cache_dir(path)
    shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    command = ["demucs", "--two-stems=vocals", "-o", str(cache_root), str(path)]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=DEMUCS_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    vocals_path = find_demucs_vocals(cache_root)
    if vocals_path is None:
        return None

    vocal_pcm = decode_wave_pcm(vocals_path)
    if vocal_pcm is None:
        return None
    vocal_samples, sample_rate = vocal_pcm
    mix_samples = decode_track_pcm(path, sample_rate, VOCALNESS_SECONDS)
    if mix_samples is None:
        return None
    return vocalness_from_stem_ratio(vocal_samples, mix_samples)


def demucs_track_cache_dir(path: Path) -> Path:
    """Return the local cache directory for one Demucs separation run."""

    key = sha256(str(path.expanduser().resolve()).encode("utf-8")).hexdigest()[:16]
    return config.ensure_data_dir() / "cache" / "models" / "demucs" / key


def find_demucs_vocals(cache_root: Path) -> Path | None:
    """Find the vocals stem written by Demucs under one cache directory."""

    return find_vocal_stem(cache_root)


def find_vocal_stem(cache_root: Path) -> Path | None:
    """Find a vocal stem under a separator output directory."""

    exact = sorted(cache_root.rglob("vocals.wav"))
    if exact:
        return exact[0]
    candidates = sorted(path for path in cache_root.rglob("*.wav") if "vocal" in path.name.lower())
    return candidates[0] if candidates else None


def decode_track_pcm(path: Path, sample_rate: int, seconds: int) -> list[int] | None:
    """Decode a local track to PCM samples without assuming its container format."""

    if path.suffix.lower() == ".wav":
        wave_pcm = decode_wave_pcm(path)
        if wave_pcm is not None:
            samples, _sample_rate = wave_pcm
            return samples[: sample_rate * seconds]
    return decode_pcm_with_ffmpeg(path, sample_rate, seconds)


def vocalness_from_stem_ratio(vocal_samples: list[int], mix_samples: list[int]) -> float | None:
    """Estimate vocalness from the vocal stem RMS relative to the original mix RMS."""

    vocal_rms = rms(vocal_samples)
    mix_rms = rms(mix_samples)
    if vocal_rms is None or mix_rms is None or mix_rms <= 0.0:
        return None
    return round(min(max(vocal_rms / mix_rms, 0.0), 1.0), 2)


def rms(samples: list[int]) -> float | None:
    """Return root mean square amplitude for integer PCM samples."""

    if not samples:
        return None
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


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
