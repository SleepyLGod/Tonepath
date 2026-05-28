"""Local embedding cache helpers for optional music-text experiments."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from tonepath import config
from tonepath.model_runtime import CLAP_MODEL_ID, run_clap_audio_embedding, run_clap_text_embedding, run_clap_text_embeddings
from tonepath.models import Track


def clap_embedding_root() -> Path:
    """Return the workspace-local CLAP embedding cache root."""

    return config.ensure_data_dir() / "cache" / "embeddings" / "clap"


def clap_audio_embedding_path(track: Track) -> Path:
    """Return the cache path for one track's CLAP audio embedding."""

    if track.id is None:
        raise ValueError("Track must be persisted before embedding analysis.")
    return clap_embedding_root() / "audio" / f"{track.id}.json"


def clap_text_embedding_path(text: str, model_id: str = CLAP_MODEL_ID) -> Path:
    """Return the cache path for one CLAP text embedding."""

    digest = sha256(f"{model_id}\n{text}".encode("utf-8")).hexdigest()
    return clap_embedding_root() / "text" / f"{digest}.json"


def track_embedding_metadata(track: Track, dimension: int, model_id: str = CLAP_MODEL_ID) -> dict[str, object]:
    """Return cache metadata for one track embedding."""

    stat = track.path.stat()
    return {
        "track_id": track.id,
        "model_id": model_id,
        "dimension": dimension,
        "file_size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def embedding_cache_key(metadata: dict[str, object]) -> str:
    """Return a stable cache key for embedding metadata."""

    fields = (
        metadata.get("track_id"),
        metadata.get("file_size"),
        metadata.get("mtime"),
        metadata.get("model_id"),
        metadata.get("dimension"),
    )
    return sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


def read_clap_audio_embedding(track: Track) -> list[float] | None:
    """Return a cached CLAP audio embedding if it still matches the track."""

    path = clap_audio_embedding_path(track)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    embedding = vector_from_payload(payload)
    if embedding is None:
        return None
    expected = track_embedding_metadata(track, len(embedding), str(payload.get("model_id") or CLAP_MODEL_ID))
    if payload.get("cache_key") != embedding_cache_key(expected):
        return None
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    return embedding


def write_clap_audio_embedding(track: Track, payload: dict[str, object]) -> Path:
    """Persist one CLAP audio embedding cache file."""

    embedding = vector_from_payload(payload)
    if embedding is None:
        raise RuntimeError("CLAP audio embedding payload is missing a numeric embedding.")
    model_id = str(payload.get("model_id") or CLAP_MODEL_ID)
    metadata = track_embedding_metadata(track, len(embedding), model_id)
    output = {
        **metadata,
        "cache_key": embedding_cache_key(metadata),
        "embedding": embedding,
    }
    path = clap_audio_embedding_path(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def clap_audio_embedding_missing(track: Track) -> bool:
    """Return whether one track needs CLAP audio embedding analysis."""

    return read_clap_audio_embedding(track) is None


def read_or_create_clap_text_embedding(text: str) -> list[float]:
    """Return a cached CLAP text embedding, creating it when missing."""

    path = clap_text_embedding_path(text)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("text") == text and payload.get("model_id") == CLAP_MODEL_ID:
            embedding = vector_from_payload(payload)
            if embedding is not None:
                return embedding
    payload = run_clap_text_embedding(text)
    embedding = vector_from_payload(payload)
    if embedding is None:
        raise RuntimeError("CLAP text embedding payload is missing a numeric embedding.")
    output = {
        "text": text,
        "model_id": str(payload.get("model_id") or CLAP_MODEL_ID),
        "dimension": len(embedding),
        "embedding": embedding,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return embedding


def read_or_create_clap_text_embeddings(texts: Iterable[str]) -> dict[str, list[float]]:
    """Return cached CLAP text embeddings for multiple probe strings."""

    unique = list(dict.fromkeys(texts))
    results: dict[str, list[float]] = {}
    missing: list[str] = []
    for text in unique:
        cached = read_clap_text_embedding(text)
        if cached is None:
            missing.append(text)
        else:
            results[text] = cached
    if missing:
        payload = run_clap_text_embeddings(missing)
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(missing):
            raise RuntimeError("CLAP text embedding payload has an invalid batch shape.")
        model_id = str(payload.get("model_id") or CLAP_MODEL_ID)
        for text, raw in zip(missing, raw_embeddings):
            if not isinstance(raw, list):
                raise RuntimeError("CLAP text embedding payload contains an invalid vector.")
            embedding = [float(value) for value in raw]
            write_clap_text_embedding(text, embedding, model_id)
            results[text] = embedding
    return results


def read_clap_text_embedding(text: str) -> list[float] | None:
    """Return a cached CLAP text embedding if present."""

    path = clap_text_embedding_path(text)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("text") != text or payload.get("model_id") != CLAP_MODEL_ID:
        return None
    return vector_from_payload(payload)


def write_clap_text_embedding(text: str, embedding: list[float], model_id: str = CLAP_MODEL_ID) -> Path:
    """Persist one CLAP text embedding cache file."""

    output = {
        "text": text,
        "model_id": model_id,
        "dimension": len(embedding),
        "embedding": embedding,
    }
    path = clap_text_embedding_path(text, model_id=model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def create_clap_audio_embedding(track: Track) -> Path:
    """Run CLAP for one track and persist the audio embedding."""

    return write_clap_audio_embedding(track, run_clap_audio_embedding(track.path))


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine similarity for two numeric vectors."""

    a = [float(value) for value in left]
    b = [float(value) for value in right]
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    left_norm = math.sqrt(sum(x * x for x in a))
    right_norm = math.sqrt(sum(y * y for y in b))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def vector_from_payload(payload: dict[str, object]) -> list[float] | None:
    """Return a numeric embedding vector from a JSON payload."""

    raw = payload.get("embedding")
    if not isinstance(raw, list) or not raw:
        return None
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
