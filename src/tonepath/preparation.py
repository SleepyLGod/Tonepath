"""Shared local library preparation service for CLI and TUI callers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tonepath.analysis import AnalysisProgress, analyze_library
from tonepath.db import TonepathStore
from tonepath.model_runtime import model_runtime_status, setup_essentia_tf_runtime
from tonepath.models import Track
from tonepath.readiness import LibraryStatus, library_status
from tonepath.scanner import scan_directory


@dataclass(frozen=True)
class ScanSummary:
    """Summary of one local music scan pass."""

    total: int
    scanned_dirs: int
    skipped: int
    pruned: int
    skipped_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnalysisFailure:
    """One failed analysis item from a preparation run."""

    stage: str
    track: Track
    error: str


@dataclass(frozen=True)
class PreparationOptions:
    """Explicit inputs for one local preparation run."""

    paths: tuple[Path, ...]
    mode: str = "balanced"
    limit: int | None = None
    setup_models: bool = False


@dataclass(frozen=True)
class PreparationEvent:
    """One human-readable preparation progress update."""

    stage: str
    message: str


@dataclass(frozen=True)
class PreparationResult:
    """Final result from one local preparation run."""

    scan: ScanSummary
    failures: tuple[AnalysisFailure, ...]
    status: LibraryStatus
    runtime_ready: bool
    affect_ready: bool


def resolve_prepare_mode(configured_mode: str, *, fast: bool, full: bool) -> str:
    """Resolve CLI flags and configured mode into one supported preparation mode."""

    if fast and full:
        raise ValueError("--fast and --full cannot be used together")
    if fast:
        return "fast"
    if full:
        return "full"
    mode = configured_mode.strip().lower()
    if mode not in {"fast", "balanced", "full"}:
        raise ValueError("models.mode must be one of: fast, balanced, full")
    return mode


def scan_library(store: TonepathStore, paths: tuple[Path, ...]) -> ScanSummary:
    """Scan local paths into an existing store and prune missing files."""

    total = 0
    scanned_dirs = 0
    skipped = 0
    pruned = 0
    skipped_messages: list[str] = []
    for music_dir in paths:
        try:
            tracks = scan_directory(music_dir)
        except (FileNotFoundError, NotADirectoryError) as exc:
            skipped += 1
            skipped_messages.append(f"Skipping {music_dir}: {exc}")
            continue
        for track in tracks:
            store.upsert_track(track)
        pruned += store.prune_missing_tracks_under(music_dir, {track.path for track in tracks})
        total += len(tracks)
        scanned_dirs += 1
    return ScanSummary(
        total=total,
        scanned_dirs=scanned_dirs,
        skipped=skipped,
        pruned=pruned,
        skipped_messages=tuple(skipped_messages),
    )


def run_preparation(
    options: PreparationOptions,
    *,
    on_event: Callable[[PreparationEvent], None] | None = None,
) -> PreparationResult:
    """Scan and analyze a local library without depending on CLI or TUI code."""

    mode = resolve_prepare_mode(options.mode, fast=False, full=False)
    store = TonepathStore()
    failures: list[AnalysisFailure] = []

    def emit(stage: str, message: str) -> None:
        if on_event is not None:
            on_event(PreparationEvent(stage=stage, message=message))

    def collect(stage: str) -> Callable[[AnalysisProgress], None]:
        def handle(event: AnalysisProgress) -> None:
            if event.error is not None:
                failures.append(AnalysisFailure(stage=stage, track=event.track, error=event.error))

        return handle

    try:
        emit("scan", "Prepare: scan")
        scan_summary = scan_library(store, options.paths)
        for message in scan_summary.skipped_messages:
            emit("scan", message)
        emit(
            "scan",
            f"Scanned {scan_summary.total} track(s) from {scan_summary.scanned_dirs} director(y/ies).",
        )
        if scan_summary.pruned:
            emit("scan", f"Pruned {scan_summary.pruned} missing track(s).")
        mir_analyzed, mir_skipped = analyze_library(
            store,
            features="mir",
            method="essentia",
            changed_only=True,
            limit=options.limit,
            progress=collect("MIR"),
        )
        emit("mir", f"Prepare: MIR analyzed {mir_analyzed} track(s); skipped {mir_skipped} track(s).")

        runtime = model_runtime_status()
        runtime_ready = bool(runtime.ready)
        affect_ready = bool(getattr(runtime, "affect_ready", runtime_ready))
        if mode != "fast" and (not runtime_ready or not affect_ready) and options.setup_models:
            emit("models", "Prepare: setting up workspace-local Essentia-TF runtime.")
            runtime = setup_essentia_tf_runtime()
            runtime_ready = bool(runtime.ready)
            affect_ready = bool(getattr(runtime, "affect_ready", runtime_ready))

        if mode == "fast":
            emit("tags", "Prepare: skipped TensorFlow tags (--fast).")
        elif runtime_ready:
            tag_analyzed, tag_skipped = analyze_library(
                store,
                features="tags",
                method="essentia-tf",
                changed_only=True,
                limit=options.limit,
                progress=collect("tags"),
            )
            emit("tags", f"Prepare: tags analyzed {tag_analyzed} track(s); skipped {tag_skipped} track(s).")
            if affect_ready:
                affect_analyzed, affect_skipped = analyze_library(
                    store,
                    features="affect",
                    method="essentia-tf",
                    changed_only=True,
                    limit=options.limit,
                    progress=collect("affect"),
                )
                emit(
                    "affect",
                    f"Prepare: affect analyzed {affect_analyzed} track(s); skipped {affect_skipped} track(s).",
                )
            else:
                emit(
                    "affect",
                    "Prepare: affect skipped. Run `uv run tonepath models setup essentia-tf` for emotion models.",
                )
        elif mode == "full":
            emit(
                "tags",
                "Prepare: full tagging requires Essentia-TF. "
                "Run `uv run tonepath models setup essentia-tf` or `uv run tonepath prepare --full --setup-models`.",
            )
        else:
            emit(
                "tags",
                "Prepare: tags skipped. Run `uv run tonepath models setup essentia-tf` for vocalness tagging.",
            )

        status = library_status(store)
        return PreparationResult(
            scan=scan_summary,
            failures=tuple(failures),
            status=status,
            runtime_ready=runtime_ready,
            affect_ready=affect_ready,
        )
    finally:
        store.close()
