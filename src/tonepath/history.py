"""Saved-session history, exact replay, and local export behavior."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.models import CandidateScore, SessionPhase, SessionPlan, SessionRequest, Track


@dataclass(frozen=True)
class HistorySession:
    """One saved or recorded session summary."""

    id: int
    prompt: str
    source_state: str
    target_state: str
    duration_sec: int
    started_at: str
    ended_at: str | None
    network_mode: str
    bookmark_name: str | None
    bookmarked_at: str | None
    play_count: int
    queue_count: int

    @property
    def saved(self) -> bool:
        """Return whether this session is bookmarked."""

        return self.bookmarked_at is not None


@dataclass(frozen=True)
class HistoryQueueItem:
    """One immutable queue item snapshot."""

    position: int
    track_id: int | None
    path: Path
    title: str | None
    artist: str | None
    phase_label: str
    score: float
    confidence: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class HistoryFeedback:
    """One feedback event associated with a session."""

    id: int
    track_id: int | None
    type: str
    value: str | None
    created_at: str


@dataclass(frozen=True)
class HistoryRecord:
    """Complete local history record for one session."""

    session: HistorySession
    phases: tuple[SessionPhase, ...]
    queue: tuple[HistoryQueueItem, ...]
    feedback: tuple[HistoryFeedback, ...]


@dataclass(frozen=True)
class ReplayPreparation:
    """Resolved exact replay candidates plus missing snapshot items."""

    source_session_id: int
    plan: SessionPlan
    candidates: tuple[CandidateScore, ...]
    omitted: tuple[HistoryQueueItem, ...]


def list_history(
    store: TonepathStore,
    include_all: bool = False,
    saved_only: bool = False,
) -> list[HistorySession]:
    """List sessions visible under the requested history filter."""

    if include_all and saved_only:
        raise ValueError("--all and --saved-only cannot be used together.")
    return [
        _history_session(row)
        for row in store.history_session_rows(include_all=include_all, saved_only=saved_only)
    ]


def load_history(store: TonepathStore, session_id: int) -> HistoryRecord:
    """Load one session with its queue, phases, and feedback."""

    session_row = store.history_session_row(session_id)
    if session_row is None:
        raise RuntimeError(f"Session {session_id} was not found.")
    phases = tuple(_session_phase(row) for row in store.session_phase_rows(session_id))
    queue = tuple(_queue_item(row) for row in store.session_queue_items(session_id))
    feedback = tuple(_history_feedback(row) for row in store.session_feedback_rows(session_id))
    return HistoryRecord(
        session=_history_session(session_row),
        phases=phases,
        queue=queue,
        feedback=feedback,
    )


def prepare_replay(store: TonepathStore, session_id: int) -> ReplayPreparation:
    """Resolve an exact saved queue against files that still exist."""

    record = load_history(store, session_id)
    if not record.queue:
        raise RuntimeError(
            f"Session {session_id} does not have a saved queue snapshot. "
            "Run its original Request again to create a replayable session."
        )
    candidates, omitted = _resolve_queue(store, record)
    if not candidates:
        raise RuntimeError(f"No playable tracks remain for session {session_id}.")
    request = SessionRequest(
        prompt=record.session.prompt,
        source_state=record.session.source_state,
        target_state=record.session.target_state,
        duration_sec=record.session.duration_sec,
        no_vocals=any(phase.vocal_policy == "avoid" for phase in record.phases),
    )
    return ReplayPreparation(
        source_session_id=session_id,
        plan=SessionPlan(request=request, phases=record.phases),
        candidates=tuple(candidates),
        omitted=tuple(omitted),
    )


def create_replay_session(store: TonepathStore, replay: ReplayPreparation) -> int:
    """Persist a new session for a prepared exact replay."""

    session_id = store.save_session(replay.plan)
    store.replace_session_queue(session_id, replay.candidates)
    return session_id


def export_history_bundle(store: TonepathStore, session_id: int, output: Path) -> Path:
    """Export one session as JSON and an M3U8 playlist."""

    if output.exists():
        if not output.is_dir():
            raise RuntimeError(f"Export output is not a directory: {output}")
        if any(output.iterdir()):
            raise RuntimeError(f"Export output directory is not empty: {output}")
    else:
        output.mkdir(parents=True)

    record = load_history(store, session_id)
    candidates, omitted = _resolve_queue(store, record)
    omitted_positions = {item.position for item in omitted}
    payload = {
        "format": "tonepath-session-export-v1",
        "privacy": {
            "contains_local_paths": True,
            "local_only": True,
            "note": "This bundle contains local file paths and is intended for local storage.",
        },
        "session": asdict(record.session),
        "request": {
            "prompt": record.session.prompt,
            "source_state": record.session.source_state,
            "target_state": record.session.target_state,
            "duration_sec": record.session.duration_sec,
        },
        "phases": [asdict(phase) for phase in record.phases],
        "queue": [
            {
                **_queue_item_payload(item),
                "available": item.position not in omitted_positions,
            }
            for item in record.queue
        ],
        "feedback": [asdict(item) for item in record.feedback],
        "omitted": [_queue_item_payload(item) for item in omitted],
    }
    lines = ["#EXTM3U"]
    for candidate in candidates:
        title = _m3u_metadata(candidate.track.title or candidate.track.path.stem)
        artist = _m3u_metadata(candidate.track.artist or "Unknown artist")
        lines.extend((f"#EXTINF:-1,{artist} - {title}", _m3u_path(candidate.track.path)))
    playlist_text = "\n".join(lines) + "\n"

    (output / "session.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "playlist.m3u8").write_text(playlist_text, encoding="utf-8")
    return output


def _resolve_queue(
    store: TonepathStore,
    record: HistoryRecord,
) -> tuple[list[CandidateScore], list[HistoryQueueItem]]:
    phase_by_label = {phase.label: phase for phase in record.phases}
    candidates: list[CandidateScore] = []
    omitted: list[HistoryQueueItem] = []
    for item in record.queue:
        phase = phase_by_label.get(item.phase_label)
        if phase is None:
            raise RuntimeError(
                f"Session {record.session.id} queue references missing phase {item.phase_label!r}."
            )
        current = store.get_track(item.track_id) if item.track_id is not None else None
        snapshot_path = item.path.expanduser()
        if current is not None and current.path.expanduser().exists():
            track = replace(current, title=item.title, artist=item.artist)
        elif snapshot_path.exists():
            track = Track(
                id=current.id if current is not None else None,
                path=snapshot_path,
                file_hash=current.file_hash if current is not None else "",
                mtime=snapshot_path.stat().st_mtime,
                title=item.title,
                artist=item.artist,
                album=current.album if current is not None else None,
                genre=current.genre if current is not None else None,
                duration=current.duration if current is not None else None,
                format=current.format if current is not None else snapshot_path.suffix.lstrip(".") or None,
            )
        else:
            omitted.append(item)
            continue
        candidates.append(
            CandidateScore(
                track=track,
                phase=phase,
                score=item.score,
                confidence=item.confidence,
                reasons=item.reasons,
            )
        )
    return candidates, omitted


def _history_session(row: dict[str, object]) -> HistorySession:
    return HistorySession(
        id=int(row["id"]),
        prompt=str(row["prompt"]),
        source_state=str(row["source_state"]),
        target_state=str(row["target_state"]),
        duration_sec=int(row["duration_sec"]),
        started_at=str(row["started_at"]),
        ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
        network_mode=str(row["network_mode"]),
        bookmark_name=str(row["bookmark_name"]) if row["bookmark_name"] is not None else None,
        bookmarked_at=str(row["bookmarked_at"]) if row["bookmarked_at"] is not None else None,
        play_count=int(row["play_count"]),
        queue_count=int(row["queue_count"]),
    )


def _session_phase(row: dict[str, object]) -> SessionPhase:
    return SessionPhase(
        label=str(row["label"]),
        start_sec=int(row["start_sec"]),
        end_sec=int(row["end_sec"]),
        target_arousal=float(row["target_arousal"]),
        target_valence=float(row["target_valence"]),
        target_energy=float(row["target_energy"]),
        vocal_policy=str(row["vocal_policy"]),
    )


def _queue_item(row: dict[str, object]) -> HistoryQueueItem:
    reasons = row["reasons"]
    return HistoryQueueItem(
        position=int(row["position"]),
        track_id=int(row["track_id"]) if row["track_id"] is not None else None,
        path=Path(str(row["track_path"])),
        title=str(row["title"]) if row["title"] is not None else None,
        artist=str(row["artist"]) if row["artist"] is not None else None,
        phase_label=str(row["phase_label"]),
        score=float(row["score"]),
        confidence=str(row["confidence"]),
        reasons=tuple(str(reason) for reason in reasons) if isinstance(reasons, list) else (),
    )


def _history_feedback(row: dict[str, object]) -> HistoryFeedback:
    return HistoryFeedback(
        id=int(row["id"]),
        track_id=int(row["track_id"]) if row["track_id"] is not None else None,
        type=str(row["type"]),
        value=str(row["value"]) if row["value"] is not None else None,
        created_at=str(row["created_at"]),
    )


def _queue_item_payload(item: HistoryQueueItem) -> dict[str, object]:
    return {
        "position": item.position,
        "track_id": item.track_id,
        "path": str(item.path),
        "title": item.title,
        "artist": item.artist,
        "phase": item.phase_label,
        "score": item.score,
        "confidence": item.confidence,
        "reasons": list(item.reasons),
    }


def _m3u_metadata(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")


def _m3u_path(path: Path) -> str:
    rendered = str(path)
    if "\r" in rendered or "\n" in rendered:
        raise RuntimeError("Cannot export a playlist entry with a line break in its path.")
    return rendered
