"""Structured metadata enrichment boundaries for Tonepath."""

from __future__ import annotations

from typing import Literal

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import EnrichmentRecord, Track


EnrichmentProvider = Literal["local", "musicbrainz", "acoustid", "listenbrainz", "web"]
ONLINE_PROVIDERS = {"musicbrainz", "acoustid", "listenbrainz", "web"}


def enrich_library(store: TonepathStore, provider: EnrichmentProvider, confirm: bool = False) -> int:
    """Run a source-attributed enrichment provider over the local library."""

    if provider == "local":
        return enrich_local_metadata(store)

    if provider in ONLINE_PROVIDERS:
        settings = config.load_config()
        if settings.network_mode != "online" or not confirm:
            raise PermissionError(
                f"{provider} enrichment is online-only and requires network_mode='online' plus --confirm."
            )
        raise NotImplementedError(f"{provider} enrichment is planned but not implemented; no network request was made.")

    raise ValueError(f"Unsupported enrichment provider: {provider}")


def enrich_local_metadata(store: TonepathStore) -> int:
    """Store existing local metadata as explicit local enrichment records."""

    count = 0
    for track in store.list_tracks():
        if track.id is None:
            continue
        for field, value in local_fields(track).items():
            store.upsert_enrichment(
                EnrichmentRecord(
                    track_id=track.id,
                    field=field,
                    value=value,
                    tier="local",
                    source="local-metadata",
                    confidence="medium" if field in {"title", "artist", "album", "genre"} else "low",
                    is_online=False,
                )
            )
            count += 1
    return count


def local_fields(track: Track) -> dict[str, str]:
    """Return non-empty local metadata fields suitable for enrichment storage."""

    fields: dict[str, str] = {}
    if track.title:
        fields["title"] = track.title
    if track.artist:
        fields["artist"] = track.artist
    if track.album:
        fields["album"] = track.album
    if track.genre:
        fields["genre"] = track.genre
    if track.format:
        fields["format"] = track.format
    if track.duration is not None:
        fields["duration_sec"] = f"{track.duration:.0f}"
    return fields
