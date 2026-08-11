"""Tonepath command-line interface."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

try:
    import typer
    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise RuntimeError("Tonepath CLI dependencies are missing. Run `uv sync` first.") from exc

from tonepath.analysis import AnalysisProgress, analyze_library
from tonepath.db import TonepathStore
from tonepath.display import clean_metadata_text, display_artist, display_label, display_title
from tonepath.doctor import run_doctor
from tonepath.enrichment import EnrichmentProvider, enrich_library
from tonepath.experience import listen_intelligence_summary, setup_next_step, smart_plan_session
from tonepath.evaluation import (
    evaluate_audit,
    evaluate_bakeoff,
    evaluate_diagnose,
    evaluate_intent,
    evaluate_profile_comparison,
    evaluate_rerank,
    evaluate_selection,
    evaluate_suite,
    run_codex_audit,
)
from tonepath.explanation import explain_candidate
from tonepath.history import (
    create_replay_session,
    export_history_bundle,
    list_history,
    load_history,
    prepare_replay,
)
from tonepath.llm import llm_doctor, parse_prompt_with_llm, provider_config
from tonepath.library import clear_metadata_override, library_issues, set_metadata_override
from tonepath.memory import (
    add_memory_log,
    build_memory_evidence,
    consolidate_memory_with_llm,
    ensure_memory_profile,
    memory_profile_path,
    memory_profile_text,
    memory_suggestions_from_llm,
    save_consolidated_memory_profile,
    write_memory_evidence,
)
from tonepath.model_runtime import (
    isolation_report,
    model_runtime_report,
    model_runtime_status,
    setup_clap_runtime,
    setup_essentia_tf_runtime,
)
from tonepath.models import CandidateScore, SessionPlan, Track
from tonepath.planner import plan_session, request_constraints
from tonepath.playback import MpvAdapter
from tonepath.playback_controller import PlaybackController
from tonepath import preparation as tonepath_preparation
from tonepath.preparation import AnalysisFailure, PreparationEvent, PreparationOptions, ScanSummary
from tonepath.profile import (
    active_rule_payload,
    apply_suggestion_group,
    apply_suggestion,
    build_profile_evidence,
    deterministic_suggestions,
    list_pending_suggestions,
    memory_context_text,
    profile_evidence_latest_path,
    profile_learning_hint,
    profile_memory_path,
    profile_readiness,
    pending_suggestion_groups,
    run_codex_profile_suggest,
    save_suggestions,
    suggest_with_llm,
    write_profile_evidence,
    write_profile_evidence_markdown,
    write_profile_memory,
)
from tonepath.privacy import (
    ALL_PERSONAL_CATEGORIES,
    build_privacy_inventory,
    execute_privacy_delete,
    export_personal_data,
    plan_privacy_delete,
    privacy_status,
    render_delete_plan,
    render_delete_result,
    render_privacy_inventory,
)
from tonepath.readiness import (
    LibraryStatus,
    library_status,
    quality_check_hint,
    readiness_blocks_session,
    readiness_label,
    status_next_action,
)
from tonepath.selector import select_path
from tonepath.setup import SetupDraft, setup_review, setup_summary, validate_music_directories
from tonepath.tui import run_tui
from tonepath import config as tonepath_config


app = typer.Typer(help="Local-first music state-transition agent.")
config_app = typer.Typer(help="Manage local config.")
feedback_app = typer.Typer(help="Record local feedback.")
profile_app = typer.Typer(help="Inspect, export, or delete local profile data.")
profile_memory_app = typer.Typer(help="Manage human-editable profile memory.")
profile_evidence_app = typer.Typer(help="Manage human-readable profile evidence.")
privacy_app = typer.Typer(help="Inspect local privacy status.")
explain_app = typer.Typer(help="Explain selections.")
eval_app = typer.Typer(help="Evaluate local selection quality.")
models_app = typer.Typer(help="Manage local model runtimes.")
models_setup_app = typer.Typer(help="Set up local model runtimes.")
llm_app = typer.Typer(help="Inspect optional LLM integrations.")
library_app = typer.Typer(help="Inspect library hygiene and local metadata overrides.")
memory_app = typer.Typer(help="Manage private memory notes and consolidated listening profile.")
history_app = typer.Typer(help="Inspect, save, replay, and export listening sessions.")

app.add_typer(config_app, name="config")
app.add_typer(feedback_app, name="feedback")
app.add_typer(profile_app, name="profile")
profile_app.add_typer(profile_memory_app, name="memory")
profile_app.add_typer(profile_evidence_app, name="evidence")
app.add_typer(privacy_app, name="privacy")
app.add_typer(explain_app, name="explain")
app.add_typer(eval_app, name="eval")
app.add_typer(models_app, name="models")
models_app.add_typer(models_setup_app, name="setup")
app.add_typer(llm_app, name="llm")
app.add_typer(library_app, name="library")
app.add_typer(memory_app, name="memory")
app.add_typer(history_app, name="history")

console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Open the TUI when no subcommand is provided."""

    if ctx.invoked_subcommand is None:
        run_tui()


@app.command("tui")
def tui_command(
    prompt: Annotated[str | None, typer.Argument(help="Optional state transition prompt for the TUI.")] = None,
) -> None:
    """Open the controlled Textual session screen."""

    if prompt is None:
        run_tui()
        return
    run_tui(prompt=prompt)


@app.command()
def setup(
    preset: Annotated[str | None, typer.Option("--preset", help="Experience preset: private, smart, or custom.")] = None,
    music_dir: Annotated[Path | None, typer.Option("--music-dir", help="Music directory for scripted --preset setup.")] = None,
    allow_model_setup: Annotated[bool | None, typer.Option("--allow-model-setup/--no-allow-model-setup", help="Allow local model setup for scripted --preset setup.")] = None,
    send_to_llm: Annotated[bool | None, typer.Option("--send-to-llm/--no-send-to-llm", help="Allow external text processing for scripted --preset setup.")] = None,
    llm_provider: Annotated[str | None, typer.Option("--llm-provider", help="AI provider for scripted --preset setup; no key is stored.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the config that would be written.")] = False,
) -> None:
    """Configure Tonepath through guided setup or a scriptable preset."""

    if preset is None:
        scripted_options = [
            name
            for name, value in (
                ("--music-dir", music_dir),
                ("--allow-model-setup", allow_model_setup),
                ("--send-to-llm", send_to_llm),
                ("--llm-provider", llm_provider),
            )
            if value is not None
        ]
        if scripted_options:
            names = ", ".join(scripted_options)
            noun = "Option" if len(scripted_options) == 1 else "Options"
            verb = "requires" if len(scripted_options) == 1 else "require"
            raise typer.BadParameter(f"{noun} {names} {verb} --preset private, smart, or custom")
        run_guided_setup(dry_run=dry_run)
        return
    try:
        settings = tonepath_config.preset_config(
            preset,
            music_dir=music_dir,
            allow_model_setup=allow_model_setup,
            send_to_llm=send_to_llm,
            llm_provider=llm_provider,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    rendered = tonepath_config.render_config(settings)
    if dry_run:
        console.print(f"Would write config: {tonepath_config.config_path()}")
        console.print(rendered, markup=False, end="")
        return
    path = tonepath_config.write_config(settings)
    console.print(f"Configured Tonepath {settings.experience.mode.title()} experience: {path}")
    console.print(setup_next_step(settings))


def run_guided_setup(*, dry_run: bool) -> None:
    """Run first-time or selective interactive setup without writing before review."""

    current = tonepath_config.load_config()
    first_run = not tonepath_config.config_path().exists()
    draft = SetupDraft.from_config(current)
    prepare_requested = False
    if first_run:
        console.print("[bold]Getting Started[/bold]")
        console.print("Step 1 of 3 · Music")
        default_dir = draft.music_dirs[0] if draft.music_dirs else "~/Music"
        while True:
            music_value = typer.prompt("Music directory", default=default_dir)
            try:
                candidate = draft.replace_music_dirs((music_value,))
                validate_music_directories(candidate.music_dirs)
            except ValueError as exc:
                console.print(str(exc))
                continue
            draft = candidate
            break
        console.print("Step 2 of 3 · Experience")
        draft = prompt_experience(draft)
    else:
        console.print(setup_summary_for_draft(draft))
        draft, prepare_requested = prompt_setup_sections(draft)

    console.print("Step 3 of 3 · Review & Start" if first_run else "Review changes")
    console.print(setup_review_for_draft(draft))
    if not typer.confirm("Save this setup?", default=False):
        console.print("Setup cancelled; no changes were saved.")
        return

    settings = draft.to_config(current)
    rendered = tonepath_config.render_config(settings)
    if dry_run:
        console.print(f"Would write config: {tonepath_config.config_path()}")
        console.print(rendered, markup=False, end="")
        return

    path = tonepath_config.write_config(settings)
    console.print(f"Configured Tonepath {settings.experience.mode.title()} experience: {path}")
    prepare_now = typer.confirm("Prepare library now?", default=prepare_requested)
    if not prepare_now:
        console.print(setup_next_step(settings))
        return

    runtime = model_runtime_status()
    setup_models_now = False
    if not bool(runtime.ready) or not bool(getattr(runtime, "affect_ready", runtime.ready)):
        setup_models_now = typer.confirm(
            "Set up optional local models for tags and emotion evidence?",
            default=False,
        )
    try:
        run_cli_preparation(
            settings,
            limit=None,
            mode=settings.models.mode,
            setup_models=setup_models_now,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"Setup was saved, but preparation failed: {exc}")
        console.print("Retry with `uv run tonepath prepare`.")
        raise typer.Exit(code=1) from exc


def prompt_setup_sections(draft: SetupDraft) -> tuple[SetupDraft, bool]:
    """Let an existing user edit only selected setup sections."""

    prepare_requested = False
    while True:
        section = prompt_choice(
            "Change (music/experience/models/local-data/prepare/done)",
            {"music", "experience", "models", "local-data", "prepare", "done"},
            default="done",
        )
        if section == "done":
            return draft, prepare_requested
        if section == "music":
            draft = prompt_music_directories(draft)
        elif section == "experience":
            draft = prompt_experience(draft)
        elif section == "models":
            mode = prompt_choice("Analysis mode (fast/balanced/full)", {"fast", "balanced", "full"}, draft.model_mode)
            allow_setup = typer.confirm("Allow explicit local model setup when requested?", default=draft.allow_model_setup)
            draft = draft.with_models(mode, allow_setup=allow_setup)
        elif section == "local-data":
            history = typer.confirm("Store playback history locally?", default=draft.store_play_history)
            draft = draft.with_local_history(history)
            if draft.experience_mode != "private":
                consent = typer.confirm(
                    "Allow Request and Memory text to be sent for opted-in AI tasks?",
                    default=draft.send_to_llm,
                )
                draft = draft.with_experience(
                    draft.experience_mode,
                    send_to_llm=consent,
                    provider=draft.llm_provider,
                )
        else:
            prepare_requested = True


def prompt_music_directories(draft: SetupDraft) -> SetupDraft:
    """Add or remove one music directory without silently replacing the list."""

    console.print("Music directories:")
    for path in draft.music_dirs:
        console.print(f"- {path}")
    operation = prompt_choice("Music action (keep/add/remove)", {"keep", "add", "remove"}, "keep")
    if operation == "keep":
        return draft
    if operation == "add":
        while True:
            raw_path = typer.prompt("Music directory")
            try:
                validate_music_directories((raw_path,))
            except ValueError as exc:
                console.print(str(exc), markup=False)
                continue
            return draft.add_music_dir(Path(raw_path))
    raw_path = typer.prompt("Music directory")
    updated = draft.remove_music_dir(Path(raw_path))
    if updated.music_dirs == draft.music_dirs:
        console.print(f"No matching music directory was removed: {Path(raw_path).expanduser()}", markup=False)
        return draft
    try:
        validate_music_directories(updated.music_dirs)
    except ValueError as exc:
        console.print(str(exc), markup=False)
        return draft
    return updated


def prompt_experience(draft: SetupDraft) -> SetupDraft:
    """Collect one simple experience choice and explicit external-text consent."""

    mode = prompt_choice("Experience (private/smart/custom)", tonepath_config.EXPERIENCE_PRESETS, draft.experience_mode)
    if mode == "private":
        return draft.with_experience("private", send_to_llm=False, provider=draft.llm_provider)
    if mode == "smart":
        consent = typer.confirm("Allow Request and Memory text to use AI Assist?", default=False)
        return draft.with_experience("smart", send_to_llm=consent, provider=draft.llm_provider)

    model_mode = prompt_choice("Analysis mode (fast/balanced/full)", {"fast", "balanced", "full"}, draft.model_mode)
    provider = prompt_choice("AI provider (deepseek/qwen)", tonepath_config.LLM_PROVIDERS, draft.llm_provider)
    consent = typer.confirm("Allow Request and Memory text to use AI Assist?", default=draft.send_to_llm)
    history = typer.confirm("Store playback history locally?", default=draft.store_play_history)
    allow_setup = typer.confirm("Allow explicit local model setup when requested?", default=draft.allow_model_setup)
    return (
        draft.with_experience("custom", send_to_llm=consent, provider=provider)
        .with_models(model_mode, allow_setup=allow_setup)
        .with_local_history(history)
    )


def prompt_choice(label: str, choices: set[str], default: str) -> str:
    """Prompt until the user enters one supported lowercase choice."""

    while True:
        value = typer.prompt(label, default=default).strip().lower()
        if value in choices:
            return value
        console.print(f"Choose one of: {', '.join(sorted(choices))}")


def setup_review_for_draft(draft: SetupDraft) -> str:
    """Return a setup review with current non-secret runtime readiness."""

    runtime = model_runtime_status()
    provider_ready = provider_config(draft.llm_provider).configured
    return setup_review(
        draft,
        model_ready=bool(runtime.ready) and bool(getattr(runtime, "affect_ready", runtime.ready)),
        provider_key_ready=provider_ready,
    )


def setup_summary_for_draft(draft: SetupDraft) -> str:
    """Return a current setup summary with non-secret runtime readiness."""

    runtime = model_runtime_status()
    provider_ready = provider_config(draft.llm_provider).configured
    settings = draft.to_config(tonepath_config.load_config())
    return setup_summary(
        settings,
        model_ready=bool(runtime.ready) and bool(getattr(runtime, "affect_ready", runtime.ready)),
        provider_key_ready=provider_ready,
    )


@app.command()
def listen(
    prompt: Annotated[str, typer.Argument(help="What you feel now and what music should help you become.")],
    dry_run: Annotated[bool, typer.Option(help="Print the selected path without launching mpv.")] = False,
    background: Annotated[bool, typer.Option(help="Start mpv in the background and return immediately.")] = False,
    limit_per_phase: Annotated[int, typer.Option(help="Number of tracks to select per phase.")] = 2,
) -> None:
    """Smart default entrypoint: check readiness, plan a path, and play or preview it."""

    settings = tonepath_config.load_config()
    store = TonepathStore()
    try:
        status = library_status(store)
        runtime_ready = model_runtime_status().ready
        console.print(f"Tonepath experience: {settings.experience.mode.title()}")
        console.print(listen_intelligence_summary(settings, runtime_ready))
        if status.tracks == 0:
            console.print("No prepared library yet.")
            console.print(f"Next action: {status_next_action(status, runtime_ready, settings)}")
            raise typer.Exit(code=1)
        readiness = readiness_label(status, runtime_ready, settings)
        if readiness != "Ready for TUI":
            console.print(f"Readiness: {readiness}")
            console.print(f"Guidance: {status_next_action(status, runtime_ready, settings)}")
            if readiness_blocks_session(readiness):
                raise typer.Exit(code=1)
        plan, note = smart_plan_session(prompt, settings)
        if note:
            console.print(note)
        run_planned_session(store, plan, dry_run=dry_run, background=background, limit_per_phase=limit_per_phase)
    finally:
        store.close()


@app.command()
def scan(path: Annotated[Path | None, typer.Argument(help="Optional local music directory to scan.")] = None) -> None:
    """Scan local music directories into the Tonepath library."""

    paths = resolve_scan_paths(path)
    store = TonepathStore()
    try:
        summary = tonepath_preparation.scan_library(store, paths)
    finally:
        store.close()

    for message in summary.skipped_messages:
        console.print(message)
    print_scan_summary(summary)
    if summary.skipped and summary.scanned_dirs == 0:
        raise typer.Exit(code=1)


@app.command()
def prepare(
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of eligible tracks per analysis pass.")] = None,
    fast: Annotated[bool, typer.Option("--fast", help="Skip TensorFlow tagging and only prepare scan plus MIR features.")] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Require model-backed tagging; print setup guidance if the runtime is missing."),
    ] = False,
    setup_models: Annotated[
        bool,
        typer.Option("--setup-models", help="Set up missing workspace-local model runtimes before model analysis."),
    ] = False,
) -> None:
    """Prepare local music for normal Tonepath use."""

    settings = tonepath_config.load_config()
    try:
        mode = tonepath_preparation.resolve_prepare_mode(settings.models.mode, fast=fast, full=full)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if setup_models and not settings.models.allow_setup:
        console.print(
            "Model setup is disabled by the current config. Open `uv run tonepath setup`, "
            "choose Local Models, allow explicit model setup, then rerun with `--setup-models`."
        )
        raise typer.Exit(code=1)
    try:
        run_cli_preparation(
            settings,
            limit=limit,
            mode=mode,
            setup_models=setup_models and settings.models.allow_setup,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"Preparation failed: {exc}")
        raise typer.Exit(code=1) from exc


def run_cli_preparation(
    settings: tonepath_config.TonepathConfig,
    *,
    limit: int | None,
    mode: str,
    setup_models: bool,
) -> None:
    """Run shared preparation and render its progress for CLI users."""

    def render_event(event: PreparationEvent) -> None:
        console.print(event.message)

    result = tonepath_preparation.run_preparation(
        PreparationOptions(
            paths=settings.expanded_music_dirs(),
            mode=mode,
            limit=limit,
            setup_models=setup_models,
        ),
        on_event=render_event,
    )
    print_analysis_failure_summary(list(result.failures))
    print_status_summary(result.status, runtime_ready=result.runtime_ready)


@app.command("status")
def status_command() -> None:
    """Print local library and model readiness without running analysis."""

    store = TonepathStore()
    try:
        print_status_summary(library_status(store), runtime_ready=model_runtime_status().ready)
    finally:
        store.close()


@app.command()
def start(
    prompt: Annotated[str, typer.Argument(help="State transition prompt.")],
    dry_run: Annotated[bool, typer.Option(help="Print the selected queue without launching mpv.")] = False,
    background: Annotated[bool, typer.Option(help="Start mpv in the background and return immediately.")] = False,
    limit_per_phase: Annotated[int, typer.Option(help="Number of tracks to select per phase.")] = 2,
) -> None:
    """Start a state-transition music session."""

    store = TonepathStore()
    try:
        plan = plan_session(prompt)
        run_planned_session(store, plan, dry_run=dry_run, background=background, limit_per_phase=limit_per_phase)
    finally:
        store.close()


def run_planned_session(
    store: TonepathStore,
    plan: SessionPlan,
    dry_run: bool,
    background: bool,
    limit_per_phase: int,
) -> None:
    """Create and optionally play one already-planned session."""

    candidates = select_path(store, plan, limit_per_phase=limit_per_phase)
    if not candidates:
        console.print("No tracks found. Run `tonepath scan ~/Music` first.")
        raise typer.Exit(code=1)

    render_plan(candidates)
    paths = [candidate.track.path for candidate in candidates]
    adapter = MpvAdapter()
    command = adapter.build_command(paths)
    if dry_run:
        console.print("Dry-run mpv command:")
        console.print(escape(" ".join(command)))
        console.print("Dry-run only; session not saved.")
        return

    session_id = store.save_session(plan)
    store.replace_session_queue(session_id, candidates)
    controller = PlaybackController(store, adapter=adapter)
    first_track_id = candidates[0].track.id
    process = controller.start(paths, session_id=session_id, track_id=first_track_id)
    console.print(f"Session {session_id} started with mpv PID {process.pid}.")
    if background:
        console.print("Run `tonepath stop` to stop background playback.")
        return

    try:
        controller.wait_foreground(process)
    except KeyboardInterrupt:
        console.print("Playback stopped.")
        raise typer.Exit(code=130) from None


@app.command()
def stop() -> None:
    """Stop Tonepath-managed mpv playback."""

    store = TonepathStore()
    try:
        controller = PlaybackController(store)
        pid = controller.current_pid()
        stopped = controller.stop_recorded()
        if stopped:
            console.print(f"Stopped Tonepath playback PID {pid}.")
        else:
            console.print("No active Tonepath playback. Cleared stale PID.")
    finally:
        store.close()


@app.command()
def current() -> None:
    """Print the current known session id."""

    store = TonepathStore()
    session_id = store.current_session_id()
    console.print(f"Current session: {session_id if session_id is not None else 'none'}")


@history_app.command("list")
def history_list_command(
    include_all: Annotated[bool, typer.Option("--all", help="Include sessions that were never played or saved.")] = False,
    saved_only: Annotated[bool, typer.Option("--saved-only", help="Show only bookmarked sessions.")] = False,
) -> None:
    """List local listening history."""

    store = TonepathStore()
    try:
        try:
            sessions = list_history(store, include_all=include_all, saved_only=saved_only)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if not sessions:
            console.print("No matching listening sessions.")
            return
        table = Table(title="Listening History", box=box.SIMPLE)
        table.add_column("ID", justify="right")
        table.add_column("Saved")
        table.add_column("Started")
        table.add_column("Transition")
        table.add_column("Plays", justify="right")
        table.add_column("Tracks", justify="right")
        table.add_column("Name")
        table.add_column("Request")
        for session in sessions:
            table.add_row(
                str(session.id),
                "yes" if session.saved else "",
                escape(session.started_at),
                escape(f"{session.source_state} -> {session.target_state}"),
                str(session.play_count),
                str(session.queue_count),
                escape(session.bookmark_name or ""),
                escape(session.prompt),
            )
        console.print(table)
    finally:
        store.close()


@history_app.command("show")
def history_show_command(
    session_id: Annotated[int, typer.Argument(help="Session id to inspect.")],
) -> None:
    """Show one saved session and its original queue snapshot."""

    store = TonepathStore()
    try:
        try:
            record = load_history(store, session_id)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"Session: {record.session.id}")
        console.print(f"Request: {escape(record.session.prompt)}")
        console.print(
            f"Transition: {escape(record.session.source_state)} -> "
            f"{escape(record.session.target_state)} · {record.session.duration_sec // 60}m"
        )
        console.print(
            f"Saved: {escape(record.session.bookmark_name or 'no')} · "
            f"Plays: {record.session.play_count} · Tracks: {len(record.queue)}"
        )
        rebuild_command = f"uv run tonepath listen {shlex.quote(record.session.prompt)}"
        console.print(f"Run again with new selection: {escape(rebuild_command)}")
        if not record.queue:
            console.print(
                "This legacy session does not have a queue snapshot, so exact replay is unavailable."
            )
            return
        table = Table(title="Original Queue", box=box.SIMPLE)
        table.add_column("#", justify="right")
        table.add_column("Phase")
        table.add_column("Track")
        table.add_column("Confidence")
        table.add_column("Status")
        for item in record.queue:
            label = " - ".join(part for part in (item.artist, item.title) if part) or item.path.name
            table.add_row(
                str(item.position + 1),
                escape(item.phase_label),
                escape(label),
                escape(item.confidence),
                "available" if item.path.exists() else "missing",
            )
        console.print(table)
    finally:
        store.close()


@history_app.command("save")
def history_save_command(
    session_id: Annotated[int, typer.Argument(help="Session id to bookmark.")],
    name: Annotated[str | None, typer.Option("--name", help="Optional human-readable bookmark name.")] = None,
) -> None:
    """Bookmark a listening session."""

    store = TonepathStore()
    try:
        try:
            load_history(store, session_id)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        cleaned_name = name.strip() if name and name.strip() else None
        store.save_session_bookmark(session_id, cleaned_name)
        suffix = f" as {escape(cleaned_name)}" if cleaned_name else ""
        console.print(f"Saved session {session_id}{suffix}.")
    finally:
        store.close()


@history_app.command("unsave")
def history_unsave_command(
    session_id: Annotated[int, typer.Argument(help="Session id to remove from saved sessions.")],
) -> None:
    """Remove a listening-session bookmark without deleting history."""

    store = TonepathStore()
    try:
        try:
            load_history(store, session_id)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        store.delete_session_bookmark(session_id)
        console.print(f"Session {session_id} is not saved.")
    finally:
        store.close()


@history_app.command("replay")
def history_replay_command(
    session_id: Annotated[int, typer.Argument(help="Session id to replay exactly.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without saving or launching mpv.")] = False,
    background: Annotated[bool, typer.Option("--background", help="Start mpv and return immediately.")] = False,
) -> None:
    """Replay the available files from an original saved queue."""

    store = TonepathStore()
    try:
        try:
            replay = prepare_replay(store, session_id)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        for item in replay.omitted:
            console.print(f"Omitted missing track: {escape(str(item.path))}")
        render_plan(list(replay.candidates))
        paths = [candidate.track.path for candidate in replay.candidates]
        adapter = MpvAdapter()
        if dry_run:
            console.print("Dry-run mpv command:")
            console.print(escape(" ".join(adapter.build_command(paths))))
            console.print("Dry-run only; replay session not saved.")
            return

        replay_session_id = create_replay_session(store, replay)
        controller = PlaybackController(store, adapter=adapter)
        first_track_id = replay.candidates[0].track.id
        process = controller.start(
            paths,
            session_id=replay_session_id,
            track_id=first_track_id,
        )
        console.print(
            f"Replay session {replay_session_id} started from session {session_id} "
            f"with mpv PID {process.pid}."
        )
        if background:
            console.print("Run `tonepath stop` to stop background playback.")
            return
        try:
            controller.wait_foreground(process)
        except KeyboardInterrupt:
            console.print("Playback stopped.")
            raise typer.Exit(code=130) from None
    finally:
        store.close()


@history_app.command("export")
def history_export_command(
    session_id: Annotated[int, typer.Argument(help="Session id to export.")],
    output: Annotated[Path, typer.Option("--output", help="New or empty output directory.")],
) -> None:
    """Export one session to JSON and M3U8 files."""

    store = TonepathStore()
    try:
        try:
            exported = export_history_bundle(store, session_id, output.expanduser())
        except (RuntimeError, ValueError, OSError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"Exported session {session_id} to {escape(str(exported))}.")
        console.print("This bundle contains local file paths; keep it on trusted local storage.")
    finally:
        store.close()


@app.command()
def analyze(
    features: Annotated[str, typer.Option(help="Feature tier: basic, vocalness, mir, tags, affect, or embedding.")] = "basic",
    method: Annotated[str, typer.Option(help="Analysis method: spectral, audio-separator, demucs-cli, essentia, essentia-tf, or clap.")] = "spectral",
    only_missing: Annotated[bool, typer.Option("--only-missing", help="Analyze only tracks missing the requested feature.")] = False,
    changed_only: Annotated[bool, typer.Option("--changed-only", help="Analyze only files changed since the last scan.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-analyze tracks even when existing results are present.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of eligible tracks to analyze.")] = None,
) -> None:
    """Run local audio feature analysis for scanned tracks."""

    if features not in {"basic", "vocalness", "mir", "tags", "affect", "embedding"}:
        raise typer.BadParameter("only basic, vocalness, mir, tags, affect, and embedding feature analysis are implemented")
    if features == "vocalness" and method not in {"spectral", "audio-separator", "demucs-cli"}:
        raise typer.BadParameter("only spectral, audio-separator, and demucs-cli vocalness methods are implemented")
    if features == "mir" and method != "essentia":
        raise typer.BadParameter("only essentia is supported for mir analysis")
    if features == "tags" and method not in {"essentia", "essentia-tf"}:
        raise typer.BadParameter("only essentia and essentia-tf are supported for tags analysis")
    if features == "affect" and method != "essentia-tf":
        raise typer.BadParameter("only essentia-tf is supported for affect analysis")
    if features == "embedding" and method != "clap":
        raise typer.BadParameter("only clap is supported for embedding analysis")
    if features == "basic" and method != "spectral":
        raise typer.BadParameter("--method is only supported with --features vocalness, mir, tags, affect, or embedding")
    store = TonepathStore()
    try:
        try:
            analyzed, skipped = analyze_library(
                store,
                features=features,
                method=method,
                only_missing=only_missing,
                changed_only=changed_only,
                force=force,
                limit=limit,
                progress=print_analysis_progress,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except KeyboardInterrupt as exc:
            console.print("Analysis interrupted. Completed results were kept; rerun with --only-missing to resume.")
            raise typer.Exit(code=130) from exc
    finally:
        store.close()
    console.print(f"Analyzed {analyzed} track(s); skipped {skipped} track(s).")


def print_analysis_progress(event: AnalysisProgress) -> None:
    """Print one local analysis progress event."""

    label = display_track(event.track)
    console.print(f"[{event.index}/{event.total}] analyzing: {label}")
    if event.error is not None:
        console.print(f"error: {event.error}")
        return
    if event.result is None:
        console.print("result: skipped")
        return
    energy = "unknown" if event.result.energy is None else f"{event.result.energy:.2f}"
    bpm = "unknown" if event.result.bpm is None else f"{event.result.bpm:.1f}"
    vocalness = "unknown" if event.result.vocalness is None else f"{event.result.vocalness:.2f}"
    arousal = "unknown" if event.result.arousal_estimate is None else f"{event.result.arousal_estimate:.2f}"
    valence = "unknown" if event.result.valence_estimate is None else f"{event.result.valence_estimate:.2f}"
    runtime = 0.0 if event.runtime_sec is None else event.runtime_sec
    console.print(
        f"result: energy={energy} bpm={bpm} vocalness={vocalness} arousal={arousal} valence={valence} "
        f"source={event.result.feature_source} "
        f"confidence={event.result.confidence} runtime={runtime:.1f}s"
    )


def print_analysis_failure_summary(failures: list[AnalysisFailure], limit: int = 5) -> None:
    """Print a short user-facing summary for failed prepare analysis files."""

    if not failures:
        return
    console.print("Prepare: some files could not be analyzed:")
    for failure in failures[:limit]:
        path = display_relative_path(failure.track.path)
        diagnostic = audio_probe_diagnostic(failure.track.path)
        console.print(f"- {failure.stage}: {escape(display_track(failure.track))} ({escape(path)})")
        console.print(f"  error: {escape(concise_analysis_error(failure.error))}")
        if diagnostic:
            console.print(f"  probe: {escape(diagnostic)}")
    if len(failures) > limit:
        console.print(f"... and {len(failures) - limit} more failed analysis item(s).")


def concise_analysis_error(error: str) -> str:
    """Return the user-relevant part of a possibly verbose analysis error."""

    lines = [line.strip() for line in error.splitlines() if line.strip()]
    if not lines:
        return "analysis failed"
    return lines[-1][:220]


def audio_probe_diagnostic(path: Path) -> str | None:
    """Return a concise ffprobe diagnostic for one failed audio file."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [ffprobe, "-hide_banner", "-v", "error", "-show_format", "-show_streams", str(path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode == 0:
        return "ffprobe can read the file; model analysis still failed."
    message = " ".join((completed.stderr or completed.stdout).split())
    if "Failed to find two consecutive MPEG audio frames" in message:
        return "invalid audio: no MPEG frames found"
    if "Invalid data found when processing input" in message:
        return "invalid audio data"
    return message[:180] if message else "ffprobe could not read this file"


def display_track(track: Track) -> str:
    """Return a compact track label for terminal progress output."""

    title = display_title(track)
    artist = display_artist(track)
    return f"{title} - {artist}"


def display_relative_path(path: Path) -> str:
    """Return a compact display path relative to the current directory when possible."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)


@app.command()
def doctor() -> None:
    """Check local Tonepath dependencies."""

    console.print(run_doctor())


@models_setup_app.command("essentia-tf")
def models_setup_essentia_tf() -> None:
    """Set up the workspace-local Essentia TensorFlow runtime."""

    console.print(isolation_report())
    try:
        status = setup_essentia_tf_runtime()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Essentia TensorFlow runtime ready: {status.runtime_dir}")


@models_setup_app.command("clap")
def models_setup_clap() -> None:
    """Set up the workspace-local CLAP embedding runtime."""

    console.print(isolation_report())
    try:
        status = setup_clap_runtime()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not status.ready:
        missing = ", ".join(status.missing)
        raise typer.BadParameter(f"CLAP runtime setup completed, but doctor still reports missing items: {missing}")
    console.print(f"CLAP runtime ready: {status.runtime_dir}")


@models_app.command("doctor")
def models_doctor() -> None:
    """Check local model runtime status."""

    console.print(model_runtime_report())


@llm_app.command("doctor")
def llm_doctor_command(provider: Annotated[str | None, typer.Option(help="Provider: deepseek or qwen.")] = None) -> None:
    """Check optional LLM configuration without printing secrets."""

    try:
        console.print(llm_doctor(provider))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("parse")
def parse_command(
    prompt: Annotated[str, typer.Argument(help="State transition prompt.")],
    use_llm: Annotated[bool, typer.Option("--llm", help="Use an opt-in LLM parser.")] = False,
    provider: Annotated[str | None, typer.Option(help="Provider: deepseek or qwen.")] = None,
) -> None:
    """Parse a prompt into a structured state-transition intent."""

    if not use_llm:
        plan = plan_session(prompt)
        payload = {
            "source_state": plan.request.source_state,
            "target_state": plan.request.target_state,
            "duration_min": plan.request.duration_sec // 60,
            "constraints": request_constraints(plan.request),
        }
        console.print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    try:
        payload = parse_prompt_with_llm(prompt, provider=provider)
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def enrich(
    provider: Annotated[EnrichmentProvider, typer.Option(help="Provider: local, musicbrainz, acoustid, listenbrainz, or web.")] = "local",
    confirm: Annotated[bool, typer.Option("--confirm", help="Confirm opt-in online enrichment when supported.")] = False,
) -> None:
    """Store source-attributed metadata enrichment fields."""

    store = TonepathStore()
    try:
        count = enrich_library(store, provider=provider, confirm=confirm)
    except (PermissionError, NotImplementedError) as exc:
        console.print(str(exc))
        raise typer.Exit(code=1) from exc
    finally:
        store.close()
    console.print(f"Stored {count} enrichment field(s) from provider: {provider}")


@library_app.command("issues")
def library_issues_command(
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for library hygiene checks.")] = False,
) -> None:
    """Show dirty metadata and duplicate display candidates."""

    store = TonepathStore()
    try:
        payload = library_issues(store)
    finally:
        store.close()
    if json_output:
        print_json_payload(payload)
        return
    render_library_issues(payload)


@library_app.command("set-meta")
def library_set_meta(
    track_id: Annotated[int, typer.Argument(help="Track id to override locally.")],
    title: Annotated[str | None, typer.Option("--title", help="Display title override.")] = None,
    artist: Annotated[str | None, typer.Option("--artist", help="Display artist override.")] = None,
) -> None:
    """Set a local display metadata override without editing audio tags."""

    store = TonepathStore()
    try:
        try:
            track = set_metadata_override(store, track_id, title, artist)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()
    console.print(f"Saved local metadata override for track {track_id}: {display_label(track)}")


@library_app.command("clear-meta")
def library_clear_meta(track_id: Annotated[int, typer.Argument(help="Track id to clear local metadata overrides for.")]) -> None:
    """Clear local display metadata overrides for one track."""

    store = TonepathStore()
    try:
        try:
            deleted, track = clear_metadata_override(store, track_id)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()
    console.print(f"Cleared {deleted} local metadata override field(s) for track {track_id}: {display_label(track)}")


@eval_app.command("selection")
def eval_selection(
    prompt: Annotated[str, typer.Argument(help="State transition prompt to evaluate.")],
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates to print.")] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
    with_profile: Annotated[bool, typer.Option("--with-profile", help="Evaluate using active profile rules.")] = False,
    no_profile: Annotated[bool, typer.Option("--no-profile", help="Evaluate with profile rules disabled.")] = False,
) -> None:
    """Evaluate selection candidates without playback or profile writes."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    if with_profile and no_profile:
        raise typer.BadParameter("Choose only one of --with-profile or --no-profile.")
    profile_enabled = not no_profile
    store = TonepathStore()
    try:
        payload = evaluate_selection(store, prompt, limit, profile_enabled=profile_enabled)
    finally:
        store.close()

    if json_output:
        print_json_payload(payload)
        return
    console.print(f"Profile: {'enabled' if profile_enabled else 'disabled'}")
    candidates = payload["candidates"]
    if not isinstance(candidates, list):
        raise TypeError("Selection evaluation payload must include candidate rows.")
    render_eval_table(candidates)


@eval_app.command("profile")
def eval_profile(
    prompt: Annotated[str, typer.Argument(help="State transition prompt to compare.")],
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates to compare.")] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
) -> None:
    """Compare selection with and without active profile rules."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    store = TonepathStore()
    try:
        payload = evaluate_profile_comparison(store, prompt, limit)
    finally:
        store.close()
    if json_output:
        print_json_payload(payload)
        return
    render_eval_profile(payload)


@eval_app.command("suite")
def eval_suite(
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates per prompt.")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
) -> None:
    """Run a read-only product-quality selection suite."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    store = TonepathStore()
    try:
        payload = evaluate_suite(store, limit)
    finally:
        store.close()

    if json_output:
        print_json_payload(payload)
        return
    render_eval_suite(payload)


@eval_app.command("bakeoff")
def eval_bakeoff(
    engines: Annotated[list[str] | None, typer.Option("--engine", help="Engine to compare: selector, clap, or hybrid. Repeat for multiple engines.")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates per scenario.")] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
) -> None:
    """Compare selector quality against optional experimental music-text engines."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    selected_engines = tuple(engines or ["selector", "clap"])
    store = TonepathStore()
    try:
        try:
            payload = evaluate_bakeoff(store, selected_engines, limit)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()

    if json_output:
        print_json_payload(payload)
        return
    render_eval_bakeoff(payload)


@eval_app.command("diagnose")
def eval_diagnose(
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates per scenario.")] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for diagnostics.")] = False,
) -> None:
    """Summarize why selection benchmark scenarios fail or warn."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    store = TonepathStore()
    try:
        try:
            payload = evaluate_diagnose(store, limit)
        except RuntimeError as exc:
            raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()

    if json_output:
        print_json_payload(payload)
        return
    render_eval_diagnose(payload)


@eval_app.command("intent")
def eval_intent(
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
) -> None:
    """Run the packaged bilingual prompt-intent parser corpus."""

    payload = evaluate_intent()
    if json_output:
        print_json_payload(payload)
        return
    render_eval_intent(payload)


@eval_app.command("audit")
def eval_audit(
    prompt: Annotated[str, typer.Argument(help="State transition prompt to audit.")],
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates to audit.")] = 12,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
    use_codex: Annotated[bool, typer.Option("--codex", help="Run optional Codex audit against the evidence pack.")] = False,
    web: Annotated[bool, typer.Option("--web", help="Allow Codex to use web search for audit evidence.")] = False,
) -> None:
    """Build a local audit pack and optionally ask Codex to review it."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    store = TonepathStore()
    try:
        evidence = evaluate_audit(store, prompt, limit)
    finally:
        store.close()

    if use_codex:
        try:
            payload = run_codex_audit(evidence, web=web)
        except (RuntimeError, OSError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if json_output:
            print_json_payload(payload)
            return
        render_codex_audit(payload)
        return

    if json_output:
        print_json_payload(evidence)
        return
    render_audit_pack(evidence)


@eval_app.command("rerank")
def eval_rerank(
    prompt: Annotated[str, typer.Argument(help="State transition prompt to rerank from the latest matching audit.")],
    latest: Annotated[bool, typer.Option("--latest", help="Use the newest matching Codex audit result.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
) -> None:
    """Preview advisory queue changes from a prior Codex audit."""

    if not latest:
        raise typer.BadParameter("Pass --latest to use the newest matching Codex audit result.")
    try:
        payload = evaluate_rerank(prompt)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if json_output:
        print_json_payload(payload)
        return
    if not payload.get("found"):
        console.print(str(payload["message"]))
        raise typer.Exit(code=1)
    render_eval_rerank(payload)


@config_app.command("init")
def config_init(overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite an existing config.")] = False) -> None:
    """Create a default local config file."""

    path = tonepath_config.init_config(overwrite=overwrite)
    action = "Wrote" if overwrite else "Ready"
    console.print(f"{action} config: {path}")


@config_app.command("show")
def config_show() -> None:
    """Print the active local config."""

    settings = tonepath_config.load_config()
    console.print(f"# {tonepath_config.config_path()}")
    console.print(tonepath_config.render_config(settings), markup=False, end="")


@config_app.command("add-music-dir")
def config_add_music_dir(path: Annotated[Path, typer.Argument(help="Music directory to add to config.")]) -> None:
    """Add a music directory to the local config."""

    settings = tonepath_config.add_music_dir(path)
    console.print(f"Added music directory: {path.expanduser()}")
    console.print(f"Configured music directories: {len(settings.music_dirs)}")


@feedback_app.command("like")
def feedback_like() -> None:
    """Record a like feedback event."""

    record_feedback("like")


@feedback_app.command("skip")
def feedback_skip() -> None:
    """Record a skip feedback event."""

    record_feedback("skip")


@feedback_app.command("too-loud")
def feedback_too_loud() -> None:
    """Record that the current music is too loud."""

    record_feedback("too-loud")


@feedback_app.command("too-slow")
def feedback_too_slow() -> None:
    """Record that the current music is too slow."""

    record_feedback("too-slow")


@feedback_app.command("no-vocals")
def feedback_no_vocals() -> None:
    """Record a no-vocals preference."""

    record_feedback("no-vocals")


@profile_app.command("inspect")
def profile_inspect(json_output: Annotated[bool, typer.Option("--json", help="Print raw summary as JSON-like repr.")] = False) -> None:
    """Inspect local profile data and active preference rules."""

    store = TonepathStore()
    summary = store.profile_summary()
    rules = store.list_profile_rules()
    store.close()
    memory_path = memory_profile_path()
    evidence_path = profile_evidence_latest_path()
    pending = list_pending_suggestions()
    groups = pending_suggestion_groups()
    active_rules = [active_rule_payload(rule) for rule in rules]
    readiness = profile_readiness(summary, active_rules, pending)
    if json_output:
        print_json_payload(
            {
                "readiness": readiness,
                "summary": summary,
                "active_rules": active_rules,
                # Keep the legacy "rules" alias for scripts written before active_rules existed.
                "rules": active_rules,
                "pending_suggestions": pending,
                "suggestion_groups": groups,
                "memory": {"path": str(memory_path), "exists": memory_path.exists()},
                "evidence": {"path": str(evidence_path), "exists": evidence_path.exists()},
            }
        )
        return
    console.print(f"Profile readiness: {readiness}")
    table = Table("Table", "Rows")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"Memory profile: {memory_path} ({'exists' if memory_path.exists() else 'missing'})")
    console.print(f"Profile evidence: {evidence_path} ({'exists' if evidence_path.exists() else 'missing'})")
    if active_rules:
        rule_table = Table("Scope", "Rule", "Target", "Threshold", "Weight", "Source", "Confidence", "Rationale", box=box.SIMPLE, expand=True)
        for rule in active_rules:
            rule_table.add_row(
                str(rule["scope"]),
                str(rule["rule_type"]),
                str(rule["target"]),
                str(rule.get("threshold", "--")),
                str(rule.get("weight", "--")),
                str(rule["source"]),
                str(rule["confidence"]),
                str(rule["rationale"]),
            )
        console.print(rule_table)
    else:
        console.print("No active profile rules.")
    if pending:
        pending_table = Table("Suggestion", "Rule", "Source", "Confidence", "Evidence", "Apply", "Rationale", box=box.SIMPLE, expand=True)
        for item in pending[:10]:
            pending_table.add_row(
                str(item.get("suggestion_id", "--")),
                str(item.get("rule_type", "--")),
                str(item.get("source", "--")),
                str(item.get("confidence", "--")),
                str(item.get("evidence_count", "--")),
                f"uv run tonepath profile apply {item.get('suggestion_id', '--')}",
                str(item.get("rationale", "")),
            )
        console.print(pending_table)
        first_id = pending[0].get("suggestion_id", "--")
        console.print(f"Apply a suggestion: uv run tonepath profile apply {first_id}")
        console.print(f"Suggestion rationale: {pending[0].get('rationale', '')}")
    if groups:
        group_table = Table("Group", "Scope", "Rules", "Confidence", "Evidence", "Apply", "Hint", box=box.SIMPLE, expand=True)
        for group in groups[:10]:
            rules_text = ", ".join(str(rule) for rule in group.get("rules", [])) if isinstance(group.get("rules"), list) else "--"
            group_table.add_row(
                str(group.get("group_id", "--")),
                str(group.get("scope", "--")),
                rules_text,
                str(group.get("confidence", "--")),
                str(group.get("evidence_count", "--")),
                str(group.get("apply_command", "--")),
                str(group.get("hint", "")),
            )
        console.print(group_table)


@profile_memory_app.command("write")
def profile_memory_write() -> None:
    """Write or refresh the human-editable profile memory Markdown file."""

    store = TonepathStore()
    try:
        path = write_profile_memory(store)
    finally:
        store.close()
    console.print(f"Profile memory written: {path}")


@profile_evidence_app.command("write")
def profile_evidence_write() -> None:
    """Write a human-readable profile evidence Markdown snapshot."""

    store = TonepathStore()
    try:
        evidence = build_profile_evidence(store)
        rules = store.list_profile_rules()
    finally:
        store.close()
    path = write_profile_evidence_markdown(evidence, rules=rules, pending_suggestions=list_pending_suggestions())
    console.print(f"Profile evidence written: {path}")


@profile_app.command("export")
def profile_export() -> None:
    """Print a profile summary for export."""

    store = TonepathStore()
    console.print(store.profile_summary())


@profile_app.command("suggest")
def profile_suggest(
    use_llm: Annotated[bool, typer.Option("--llm", help="Use configured DeepSeek/Qwen LLM to suggest profile rules.")] = False,
    use_codex: Annotated[bool, typer.Option("--codex", help="Use Codex to suggest profile rules from local evidence.")] = False,
    use_memory: Annotated[bool, typer.Option("--memory", help="Include editable Markdown profile memory/evidence as context.")] = False,
    provider: Annotated[str | None, typer.Option(help="LLM provider: deepseek or qwen.")] = None,
    confirm: Annotated[bool, typer.Option("--confirm", help="Confirm sending a privacy-safe evidence summary to the LLM.")] = False,
    web: Annotated[bool, typer.Option("--web", help="Allow Codex web search for public music context.")] = False,
) -> None:
    """Generate pending profile-rule suggestions without applying them."""

    if use_llm and use_codex:
        raise typer.BadParameter("Choose only one of --llm or --codex.")
    settings = tonepath_config.load_config()
    if use_llm and not confirm and not settings.privacy.send_to_llm:
        raise typer.BadParameter("profile suggest --llm requires --confirm or privacy.send_to_llm = true.")
    store = TonepathStore()
    try:
        evidence = build_profile_evidence(store)
        rules = store.list_profile_rules()
        memory_path = write_profile_memory(store) if use_memory else None
    finally:
        store.close()
    evidence_path = write_profile_evidence(evidence)
    memory_paths: list[Path] = []
    memory_context = None
    if use_memory:
        evidence_markdown_path = write_profile_evidence_markdown(evidence, rules=rules, pending_suggestions=list_pending_suggestions())
        memory_paths = [path for path in (memory_path, evidence_markdown_path) if path is not None]
        memory_context = memory_context_text(memory_paths)
    try:
        if use_llm:
            suggestions = suggest_with_llm(evidence, provider=provider, memory_context=memory_context)
            source = "llm"
        elif use_codex:
            payload = run_codex_profile_suggest(evidence_path, web=web, memory_paths=memory_paths)
            suggestions = payload["suggestions"]
            source = "codex"
        else:
            suggestions = deterministic_suggestions(evidence)
            source = "deterministic"
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    suggestions_path = save_suggestions(evidence, suggestions, source=source)
    render_profile_suggestions(suggestions, evidence_path, suggestions_path)


@profile_app.command("apply")
def profile_apply(suggestion_id: Annotated[str, typer.Argument(help="Pending profile suggestion id to apply.")]) -> None:
    """Apply one pending profile suggestion as an active local rule."""

    store = TonepathStore()
    try:
        rule = apply_suggestion(store, suggestion_id)
    finally:
        store.close()
    console.print(f"Applied profile rule: {rule.key}")


@profile_app.command("apply-group")
def profile_apply_group(group_id: Annotated[str, typer.Argument(help="Pending profile suggestion group id to apply.")]) -> None:
    """Apply every suggestion in one profile suggestion group."""

    store = TonepathStore()
    try:
        result = apply_suggestion_group(store, group_id)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()
    applied = result.get("applied", [])
    skipped = result.get("skipped", [])
    console.print(f"Profile suggestion group: {result['group_id']}")
    if isinstance(applied, list) and applied:
        console.print(f"Applied: {', '.join(str(item) for item in applied)}")
    else:
        console.print("Applied: none")
    if isinstance(skipped, list) and skipped:
        console.print(f"Skipped already active: {', '.join(str(item) for item in skipped)}")


@profile_app.command("delete")
def profile_delete(all_data: Annotated[bool, typer.Option("--all", help="Delete all profile/session/feedback data.")] = False) -> None:
    """Delete local profile/session/feedback data."""

    if not all_data:
        console.print("Pass --all to confirm profile deletion.")
        raise typer.Exit(code=1)
    store = TonepathStore()
    try:
        plan = plan_privacy_delete(("personalization", "history"), store=store)
        result = execute_privacy_delete(plan, store=store)
    finally:
        store.close()
    if result.failed:
        console.print(render_delete_result(result))
        raise typer.Exit(code=1)
    console.print("Deleted local profile rules, feedback, sessions, plays, profile Markdown/evidence, and pending suggestions.")


@memory_app.command("add")
def memory_add(
    text: Annotated[str | None, typer.Argument(help="Private memory text to append.")] = None,
    use_stdin: Annotated[bool, typer.Option("--stdin", help="Read private memory text from stdin.")] = False,
) -> None:
    """Append one private tree-hole memory entry to the local log."""

    if use_stdin and text is not None:
        raise typer.BadParameter("Use either text or --stdin, not both.")
    body = sys.stdin.read() if use_stdin else text
    if body is None or not body.strip():
        raise typer.BadParameter("Memory text is empty.")
    try:
        record = add_memory_log(body, source="cli-stdin" if use_stdin else "cli")
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Memory saved: {record['id']}")
    console.print("Run `uv run tonepath memory consolidate --llm --confirm` when you want to update the editable profile.")


@memory_app.command("show")
def memory_show() -> None:
    """Show the consolidated human-editable memory profile Markdown."""

    console.print(memory_profile_text(), markup=False)


@memory_app.command("edit")
def memory_edit() -> None:
    """Open the consolidated memory profile Markdown in the local editor."""

    path = ensure_memory_profile()
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    try:
        command = shlex.split(editor)
        if not command:
            raise OSError("empty editor command")
        subprocess.run([*command, str(path)], check=True)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        raise typer.BadParameter(f"Editor failed: {editor}") from exc
    console.print(f"Memory profile edited: {path}")


@memory_app.command("consolidate")
def memory_consolidate(
    use_llm: Annotated[bool, typer.Option("--llm", help="Use configured DeepSeek/Qwen LLM to consolidate memory logs.")] = False,
    provider: Annotated[str | None, typer.Option(help="LLM provider: deepseek or qwen.")] = None,
    confirm: Annotated[bool, typer.Option("--confirm", help="Confirm sending privacy-safe memory evidence to the LLM.")] = False,
) -> None:
    """Consolidate private memory logs into the editable memory profile Markdown."""

    if not use_llm:
        raise typer.BadParameter("memory consolidate currently requires --llm --confirm.")
    settings = tonepath_config.load_config()
    if not confirm and not settings.privacy.send_to_llm:
        raise typer.BadParameter("memory consolidate --llm requires --confirm or privacy.send_to_llm = true.")
    store = TonepathStore()
    try:
        evidence = build_memory_evidence(store)
        new_logs = evidence.get("new_memory_logs", [])
        if not isinstance(new_logs, list) or not new_logs:
            raise typer.BadParameter("No new memory logs to consolidate.")
        evidence_path = write_memory_evidence(evidence)
        profile_markdown = consolidate_memory_with_llm(evidence, provider=provider)
        profile_path = save_consolidated_memory_profile(store, evidence, profile_markdown)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        store.close()
    console.print(f"Memory evidence: {evidence_path}")
    console.print(f"Memory profile updated: {profile_path}")


@memory_app.command("suggest")
def memory_suggest(
    use_llm: Annotated[bool, typer.Option("--llm", help="Use configured DeepSeek/Qwen LLM to suggest profile rules from memory.")] = False,
    provider: Annotated[str | None, typer.Option(help="LLM provider: deepseek or qwen.")] = None,
    confirm: Annotated[bool, typer.Option("--confirm", help="Confirm sending privacy-safe memory evidence to the LLM.")] = False,
) -> None:
    """Generate pending profile-rule suggestions from the memory profile and new logs."""

    if not use_llm:
        raise typer.BadParameter("memory suggest currently requires --llm --confirm.")
    settings = tonepath_config.load_config()
    if not confirm and not settings.privacy.send_to_llm:
        raise typer.BadParameter("memory suggest --llm requires --confirm or privacy.send_to_llm = true.")
    store = TonepathStore()
    try:
        evidence = build_memory_evidence(store)
    finally:
        store.close()
    evidence_path = write_memory_evidence(evidence)
    try:
        suggestions = memory_suggestions_from_llm(evidence, provider=provider)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    suggestions_path = save_suggestions(evidence, suggestions, source="memory-llm")
    render_profile_suggestions(suggestions, evidence_path, suggestions_path)


@privacy_app.command("status")
def privacy_status_command() -> None:
    """Show local privacy status."""

    console.print(privacy_status())


@privacy_app.command("inspect")
def privacy_inspect_command(
    as_json: Annotated[bool, typer.Option("--json", help="Print the stable privacy inventory JSON contract.")] = False,
) -> None:
    """Inspect all known local data categories without creating storage."""

    inventory = build_privacy_inventory()
    if as_json:
        print_json_payload(inventory.to_payload())
        return
    console.print(render_privacy_inventory(inventory))


@privacy_app.command("export")
def privacy_export_command(
    output: Annotated[Path, typer.Option("--output", help="New or empty owner-only output directory.")],
) -> None:
    """Export a sanitized copy of personal Tonepath data."""

    try:
        exported = export_personal_data(output)
    except (RuntimeError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Exported personal data to {escape(str(exported))}.")
    console.print("The bundle is owner-only and excludes API keys, music paths, audio, SQLite, models, and caches.")


@privacy_app.command("delete")
def privacy_delete_command(
    categories: Annotated[
        list[str] | None,
        typer.Option("--category", help="Personal category to delete: memory, personalization, or history."),
    ] = None,
    all_personal: Annotated[
        bool,
        typer.Option("--all-personal", help="Delete Memory, Personalization, and History together."),
    ] = False,
    confirm: Annotated[bool, typer.Option("--confirm", help="Execute the displayed deletion plan.")] = False,
) -> None:
    """Preview or execute deletion of selected personal data categories."""

    if all_personal and categories:
        raise typer.BadParameter("Use either --all-personal or --category, not both.")
    requested = ALL_PERSONAL_CATEGORIES if all_personal else tuple(categories or ())
    try:
        plan = plan_privacy_delete(requested)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(render_delete_plan(plan))
    if not confirm:
        console.print("Preview only; nothing was deleted. Rerun with --confirm to execute this exact plan.")
        return
    try:
        result = execute_privacy_delete(plan)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(render_delete_result(result))
    if result.failed:
        raise typer.Exit(code=1)


@explain_app.command("current")
def explain_current() -> None:
    """Explain the first available candidate for the current path."""

    store = TonepathStore()
    plan = plan_session("steady focus 30m")
    candidates = select_path(store, plan, limit_per_phase=1)
    if not candidates:
        console.print("No candidate to explain. Run `tonepath scan ~/Music` first.")
        raise typer.Exit(code=1)
    console.print(explain_candidate(store, candidates[0]))


def record_feedback(feedback_type: str) -> None:
    """Record feedback against the current session."""

    store = TonepathStore()
    try:
        store.record_feedback(feedback_type, session_id=store.current_session_id())
    finally:
        store.close()
    console.print(f"Recorded feedback: {feedback_type}")
    console.print(profile_learning_hint())


def print_json_payload(payload: object) -> None:
    """Print machine-readable JSON without Rich line wrapping."""

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def resolve_scan_paths(path: Path | None) -> tuple[Path, ...]:
    """Return explicit scan path or configured music directories."""

    if path is not None:
        return (path.expanduser(),)
    return tonepath_config.load_config().expanded_music_dirs()


def print_scan_summary(summary: ScanSummary) -> None:
    """Print a scan summary."""

    directory_label = "directory" if summary.scanned_dirs == 1 else "directories"
    console.print(f"Scanned {summary.total} track(s) from {summary.scanned_dirs} {directory_label}.")
    if summary.pruned:
        console.print(f"Pruned {summary.pruned} missing track(s).")


def print_status_summary(status: LibraryStatus, runtime_ready: bool) -> None:
    """Print local library, model policy, and runtime readiness without secrets."""

    settings = tonepath_config.load_config()
    table = Table("Item", "Value", box=box.SIMPLE)
    table.add_row("Readiness", readiness_label(status, runtime_ready, settings))
    table.add_row("Music directories", "\n".join(settings.music_dirs))
    table.add_row("Tracks", str(status.tracks))
    table.add_row("Features", str(status.features))
    table.add_row("Missing features", str(status.missing_features))
    table.add_row("Vocalness coverage", f"{status.vocalness}/{status.tracks}")
    table.add_row("MIR coverage", f"{status.mir}/{status.tracks}")
    table.add_row("Tag coverage", f"{status.tags}/{status.tracks}")
    table.add_row("Affect coverage", f"{status.affect}/{status.tracks}")
    table.add_row("Dirty metadata", str(status.dirty_metadata))
    table.add_row("Duplicate candidates", str(status.duplicate_tracks))
    table.add_row("Tracks outside music dirs", str(status.tracks_outside_music_dirs))
    if status.missing_analysis_tracks:
        table.add_row("Missing analysis files", "\n".join(status.missing_analysis_tracks))
    table.add_row("Model mode", settings.models.mode)
    table.add_row("Model setup allowed", "yes" if settings.models.allow_setup else "no")
    table.add_row("Online models", "yes" if settings.models.allow_online else "no")
    table.add_row("Preferred tagger", settings.models.preferred_tagger)
    table.add_row("Separator fallback", settings.models.separator_fallback)
    table.add_row("Essentia-TF runtime", "ready" if runtime_ready else "missing")
    table.add_row("Data directory", str(tonepath_config.ensure_data_dir()))
    table.add_row("Network mode", settings.network_mode)
    table.add_row("Quality check", quality_check_hint(status))
    table.add_row("Next action", status_next_action(status, runtime_ready, settings))
    console.print(table)


def render_library_issues(payload: dict[str, object]) -> None:
    """Render library metadata and duplicate hygiene issues."""

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("Library issues payload summary must be an object.")
    console.print(
        "[bold]Library issues[/bold] · "
        f"dirty metadata {summary.get('dirty_metadata', 0)} · "
        f"duplicate groups {summary.get('duplicate_groups', 0)} · "
        f"duplicate candidates {summary.get('duplicate_candidates', 0)}"
    )
    dirty_rows = payload.get("dirty_metadata", [])
    if isinstance(dirty_rows, list) and dirty_rows:
        table = Table("Track ID", "Display", "Raw title", "Raw artist", "Path", "Override", box=box.SIMPLE, expand=True)
        for row in dirty_rows:
            if isinstance(row, dict):
                table.add_row(
                    str(row.get("track_id", "--")),
                    str(row.get("display_label", "--")),
                    str(row.get("raw_title") or "--"),
                    str(row.get("raw_artist") or "--"),
                    str(row.get("relative_path", "--")),
                    "yes" if row.get("has_override") else "no",
                )
        console.print(table)
    else:
        console.print("Dirty metadata: none")

    duplicate_groups = payload.get("duplicate_groups", [])
    if isinstance(duplicate_groups, list) and duplicate_groups:
        console.print("[bold]Duplicate candidates[/bold]")
        for index, group in enumerate(duplicate_groups, start=1):
            if not isinstance(group, list):
                continue
            table = Table(f"Group {index}", "Display", "Path", "Override", box=box.SIMPLE, expand=True)
            for row in group:
                if isinstance(row, dict):
                    table.add_row(
                        str(row.get("track_id", "--")),
                        str(row.get("display_label", "--")),
                        str(row.get("relative_path", "--")),
                        "yes" if row.get("has_override") else "no",
                    )
            console.print(table)
    else:
        console.print("Duplicate candidates: none")


def render_plan(candidates: list[CandidateScore]) -> None:
    """Render selected candidates as a table."""

    table = Table("Phase", "Track", "Artist", "Confidence", "Score")
    for candidate in candidates:
        table.add_row(
            escape(candidate.phase.label),
            escape(display_title(candidate.track)),
            escape(display_artist(candidate.track)),
            escape(candidate.confidence),
            f"{candidate.score:.2f}",
        )
    console.print(table)


def render_eval_table(rows: list[dict[str, object]]) -> None:
    """Render selection evaluation rows for manual product review."""

    if not rows:
        console.print("No candidates found. Run `tonepath scan ~/Music` first.")
        return
    table = Table("Phase", "Track", "Score", "Conf", "Features", "Reasons", box=box.SIMPLE, expand=True)
    for row in rows:
        track = row["track"]
        if not isinstance(track, dict):
            raise TypeError("Evaluation row track payload must be a dict.")
        features = row["features"]
        if not isinstance(features, dict):
            raise TypeError("Evaluation row features payload must be a dict.")
        reasons = row["reasons"]
        if not isinstance(reasons, list):
            raise TypeError("Evaluation row reasons payload must be a list.")
        table.add_row(
            str(row["phase"]),
            eval_track_label(track),
            str(row["score"]),
            str(row["confidence"]),
            eval_feature_summary(features),
            "\n".join(summarize_eval_reasons(str(reason) for reason in reasons)),
        )
    console.print(table)


def render_eval_profile(payload: dict[str, object]) -> None:
    """Render profile comparison output for manual review."""

    console.print(f"Profile comparison: {payload['prompt']}")
    console.print(str(payload["message"]))
    console.print(f"Active profile rules: {payload['active_rule_count']}")
    warnings = payload.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            console.print(f"Warning: {warning}")
    movements = payload.get("movements")
    if not isinstance(movements, list):
        raise TypeError("Profile comparison payload must include movements.")
    if not movements:
        console.print("No candidates found. Run `tonepath scan ~/Music` first.")
        return
    table = Table("With", "No", "ΔRank", "ΔScore", "Track", "Profile Reasons", box=box.SIMPLE, expand=True)
    for row in movements:
        if not isinstance(row, dict):
            raise TypeError("Profile movement row must be an object.")
        track = row.get("track")
        if not isinstance(track, dict):
            raise TypeError("Profile movement row must include track.")
        reasons = row.get("profile_reasons", [])
        reason_text = "\n".join(str(reason) for reason in reasons) if isinstance(reasons, list) and reasons else "--"
        table.add_row(
            str(row.get("rank_with_profile", "--")),
            str(row.get("rank_no_profile", "--")),
            str(row.get("rank_delta", "--")),
            str(row.get("score_delta", "--")),
            eval_track_label(track),
            reason_text,
        )
    console.print(table)


def render_eval_suite(suites: list[dict[str, object]]) -> None:
    """Render product-quality suite output for manual review."""

    if not suites:
        console.print("No evaluation prompts configured.")
        return
    for suite in suites:
        prompt = str(suite["prompt"])
        red_flag_count = int(suite["red_flag_count"])
        yellow_flag_count = int(suite.get("yellow_flag_count", 0))
        result = str(suite.get("result", "WARN"))
        scenario = str(suite.get("scenario_id", "ad_hoc"))
        console.print(f"\nScenario: {scenario} · {result}")
        console.print(f"Prompt: {prompt}")
        console.print(
            f"Target: {suite['source_state']} -> {suite['target_state']} · "
            f"red flags: {red_flag_count} · warnings: {yellow_flag_count} · "
            f"dirty metadata: {suite.get('dirty_metadata_count', 0)} · "
            f"duplicates: {suite.get('duplicate_candidate_count', 0)}"
        )
        checks = suite.get("checks", [])
        if isinstance(checks, list):
            console.print(f"Checks: {benchmark_check_summary(checks)}")
        candidates = suite["candidates"]
        if not isinstance(candidates, list):
            raise TypeError("Evaluation suite candidates must be a list.")
        render_eval_suite_candidates(candidates)


def render_eval_suite_candidates(rows: list[dict[str, object]]) -> None:
    """Render candidates with red flags for one suite prompt."""

    table = Table("Rank", "Phase", "Track", "Score", "Conf", "Features", "Flags", box=box.SIMPLE, expand=True)
    for index, row in enumerate(rows, start=1):
        track = row["track"]
        features = row["features"]
        if not isinstance(track, dict) or not isinstance(features, dict):
            raise TypeError("Evaluation suite row has an invalid shape.")
        flags = audit_flags(row)
        table.add_row(
            str(index),
            str(row["phase"]),
            eval_track_label(track),
            str(row["score"]),
            str(row["confidence"]),
            eval_feature_summary(features),
            "\n".join(flags) if flags else "ok",
        )
    console.print(table)


def render_eval_bakeoff(payload: list[dict[str, object]]) -> None:
    """Render music-text bake-off results."""

    for scenario in payload:
        delta = scenario.get("delta", {})
        if not isinstance(delta, dict):
            delta = {}
        console.print(
            f"\n[bold]{scenario['scenario_id']}[/bold] · {scenario['prompt']} · "
            f"verdict: {delta.get('verdict', 'inconclusive')}"
        )
        deltas = scenario.get("deltas")
        if isinstance(deltas, list):
            for row in deltas:
                if not isinstance(row, dict):
                    continue
                console.print(
                    f"  {row.get('baseline_engine', 'selector')} -> {row.get('compared_engine', '--')}: "
                    f"{row.get('verdict', 'inconclusive')}"
                )
        engines = scenario.get("engines")
        if not isinstance(engines, list):
            raise TypeError("Bake-off payload engines must be a list.")
        table = Table("Engine", "Result", "Checks", "Top candidates", box=box.SIMPLE, expand=True)
        for engine in engines:
            if not isinstance(engine, dict):
                raise TypeError("Bake-off engine rows must be objects.")
            candidates = engine.get("candidates")
            if not isinstance(candidates, list):
                raise TypeError("Bake-off engine row must include candidates.")
            top = "\n".join(
                f"{index}. {eval_track_label(row['track'])}"
                for index, row in enumerate(candidates[:3], start=1)
                if isinstance(row, dict) and isinstance(row.get("track"), dict)
            )
            table.add_row(
                str(engine["engine"]),
                str(engine["result"]),
                benchmark_check_summary(engine.get("checks", [])),
                top or "--",
            )
        console.print(table)


def render_eval_diagnose(payload: dict[str, object]) -> None:
    """Render compact evaluation diagnostics."""

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise TypeError("Diagnose payload summary must be an object.")
    counts = summary.get("result_counts", {})
    if not isinstance(counts, dict):
        counts = {}
    console.print(
        "[bold]Evaluation diagnosis[/bold] · "
        f"PASS {counts.get('PASS', 0)} · WARN {counts.get('WARN', 0)} · FAIL {counts.get('FAIL', 0)}"
    )
    top_causes = summary.get("top_root_causes", [])
    if isinstance(top_causes, list) and top_causes:
        cause_text = ", ".join(
            f"{row.get('cause')} ({row.get('count')})"
            for row in top_causes
            if isinstance(row, dict)
        )
        console.print(f"Top root causes: {cause_text}")
    console.print(f"Recommended next action: {summary.get('recommended_next_action', 'No blocking diagnosis found.')}")

    scenarios = payload.get("scenarios", [])
    if not isinstance(scenarios, list):
        raise TypeError("Diagnose payload scenarios must be a list.")
    table = Table("Scenario", "Result", "Root causes", "Next action", box=box.SIMPLE, expand=True)
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        if scenario.get("overall_result") == "PASS" and not scenario.get("root_causes"):
            continue
        table.add_row(
            str(scenario.get("scenario_id", "--")),
            str(scenario.get("overall_result", "--")),
            ", ".join(str(cause) for cause in scenario.get("root_causes", [])) or "--",
            str(scenario.get("next_action", "--")),
        )
    console.print(table)


def benchmark_check_summary(checks: list[object]) -> str:
    """Return compact benchmark check status counts."""

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status", "pass")).lower()
        if status in counts:
            counts[status] += 1
    return f"pass {counts['pass']} · warn {counts['warn']} · fail {counts['fail']}"


def render_eval_intent(payload: dict[str, object]) -> None:
    """Render bilingual intent parser evaluation output."""

    console.print(
        f"Intent fixtures: total {payload['total']} · passed {payload['passed']} · failed {payload['failed']}"
    )
    failures = payload.get("failures", [])
    if not isinstance(failures, list) or not failures:
        return
    table = Table("Lang", "Prompt", "Expected", "Actual", box=box.SIMPLE, expand=True)
    for failure in failures:
        if not isinstance(failure, dict):
            raise TypeError("Intent failure rows must be objects.")
        table.add_row(
            str(failure.get("lang", "unknown")),
            str(failure["prompt"]),
            json.dumps(failure["expected"], ensure_ascii=False),
            json.dumps(failure["actual"], ensure_ascii=False),
        )
    console.print(table)


def render_audit_pack(evidence: dict[str, object]) -> None:
    """Render local audit evidence for review before optional Codex use."""

    console.print(f"Audit evidence: {evidence['prompt']}")
    console.print(f"Evidence path: {evidence['evidence_path']}")
    candidates = evidence["candidates"]
    if not isinstance(candidates, list):
        raise TypeError("Audit evidence candidates must be a list.")
    table = Table("Rank", "Phase", "Track", "Score", "Features", "Flags", box=box.SIMPLE, expand=True)
    for index, row in enumerate(candidates, start=1):
        track = row["track"]
        features = row["features"]
        if not isinstance(track, dict) or not isinstance(features, dict):
            raise TypeError("Audit evidence row has an invalid shape.")
        flags = audit_flags(row)
        table.add_row(
            str(index),
            str(row["phase"]),
            eval_track_label(track),
            str(row["score"]),
            eval_feature_summary(features),
            "\n".join(flags) if flags else "ok",
        )
    console.print(table)


def render_codex_audit(payload: dict[str, object]) -> None:
    """Render Codex audit decisions."""

    codex = payload["codex"]
    if not isinstance(codex, dict):
        raise TypeError("Codex audit payload must include a codex object.")
    decisions = codex.get("decisions")
    if not isinstance(decisions, list):
        raise TypeError("Codex audit decisions must be a list.")
    console.print(f"Codex audit result: {payload['codex_result_path']}")
    table = Table("Track", "Decision", "Fit", "Evidence", "Reason", box=box.SIMPLE, expand=True)
    for item in decisions:
        if not isinstance(item, dict):
            raise TypeError("Codex audit decision must be an object.")
        evidence_used = item.get("evidence_used", [])
        evidence_count = len(evidence_used) if isinstance(evidence_used, list) else 0
        table.add_row(
            str(item.get("track_id")),
            str(item.get("decision")),
            str(item.get("fit_score", "--")),
            str(evidence_count),
            str(item.get("reason", "")),
        )
    console.print(table)


def render_eval_rerank(payload: dict[str, object]) -> None:
    """Render advisory rerank guidance from a matching Codex audit result."""

    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise TypeError("Rerank payload must include counts.")
    console.print(f"Rerank preview: {payload['prompt']}")
    console.print(
        f"keep {counts.get('keep', 0)} · demote {counts.get('demote', 0)} · "
        f"reject {counts.get('reject', 0)} · not audited {counts.get('not_audited', 0)}"
    )
    console.print(f"Codex result: {payload['codex_result_path']}")
    details = payload.get("details")
    if not isinstance(details, list):
        raise TypeError("Rerank payload must include details.")
    table = Table("Suggested", "Original", "Decision", "Track", "Action", "Risk / Reason", box=box.SIMPLE, expand=True)
    for row in details:
        if not isinstance(row, dict):
            raise TypeError("Rerank detail must be an object.")
        track = row.get("track")
        if not isinstance(track, dict):
            raise TypeError("Rerank detail must include track.")
        risk_flags = row.get("risk_flags", [])
        risks = ", ".join(str(flag) for flag in risk_flags) if isinstance(risk_flags, list) and risk_flags else str(row["reason"])
        table.add_row(
            str(row.get("suggested_rank", "--")),
            str(row["original_rank"]),
            str(row["decision"]).replace("_", " "),
            eval_track_label(track),
            str(row["suggested_action"]),
            risks,
        )
    console.print(table)


def render_profile_suggestions(suggestions: list[dict[str, object]], evidence_path: Path, suggestions_path: Path) -> None:
    """Render pending profile suggestions."""

    console.print(f"Profile evidence: {evidence_path}")
    console.print(f"Pending suggestions: {suggestions_path}")
    if not suggestions:
        console.print("No profile suggestions yet; more feedback is needed.")
        return
    table = Table("ID", "Scope", "Rule", "Confidence", "Rationale", box=box.SIMPLE, expand=True)
    for item in suggestions:
        table.add_row(
            str(item["suggestion_id"]),
            str(item["scope"]),
            str(item["rule_type"]),
            str(item["confidence"]),
            str(item["rationale"]),
        )
    console.print(table)


def audit_flags(row: dict[str, object]) -> list[str]:
    """Return combined red and yellow audit flags for display."""

    flags: list[str] = []
    red_flags = row.get("red_flags", [])
    yellow_flags = row.get("yellow_flags", [])
    if isinstance(red_flags, list):
        flags.extend(f"RED: {flag}" for flag in red_flags)
    if isinstance(yellow_flags, list):
        flags.extend(f"WARN: {flag}" for flag in yellow_flags)
    return flags


def eval_track_label(track: dict[str, object]) -> str:
    """Return a compact track label for evaluation tables."""

    display = track.get("display_label")
    if display:
        return str(display)
    title = clean_metadata_text(str(track["title"])) if track.get("title") is not None else None
    artist = clean_metadata_text(str(track["artist"])) if track.get("artist") is not None else None
    title = title or "unknown"
    artist = artist or "unknown"
    return f"{title} - {artist}"


def eval_cell(value: object) -> str:
    """Render missing evaluation values without inventing facts."""

    return "--" if value is None else str(value)


def eval_feature_summary(features: dict[str, object]) -> str:
    """Return compact feature details for selection review."""

    affect = features.get("affect_profile")
    affect_text = "--"
    if isinstance(affect, dict) and affect:
        affect_text = ", ".join(f"{key}={value}" for key, value in sorted(affect.items()) if key in {"sadness", "uplift", "calmness", "tension"})
    return (
        f"src={eval_cell(features['source'])}\n"
        f"energy={eval_cell(features['energy'])} loud={eval_cell(features['loudness'])}\n"
        f"bpm={eval_cell(features['bpm'])} vocal={eval_cell(features['vocalness'])}\n"
        f"aro={eval_cell(features.get('arousal'))} val={eval_cell(features.get('valence'))}\n"
        f"affect={affect_text}"
    )


def summarize_eval_reasons(reasons: Iterable[str]) -> list[str]:
    """Return short reason labels for narrow terminal evaluation tables."""

    labels: list[str] = []
    for reason in reasons:
        if "supports no-vocals" in reason:
            labels.append("no-vocals supported")
        elif "conflicts with no-vocals" in reason:
            labels.append("no-vocals conflict")
        elif "unknown" in reason and "no-vocals" in reason:
            labels.append("vocalness unknown")
        elif "inconclusive" in reason:
            labels.append("vocalness inconclusive")
        elif "energy feature" in reason:
            labels.append("energy fit")
        elif "loudness feature" in reason:
            labels.append("loudness fit")
        elif "BPM feature" in reason:
            labels.append("BPM fit")
        elif "affect profile" in reason:
            labels.append("affect fit")
        elif "sad/dark/tension" in reason:
            labels.append("affect risk")
        elif "feedback" in reason:
            labels.append("feedback adjusted")
        elif "genre" in reason:
            labels.append("genre signal")
        elif "duration is known" in reason:
            labels.append("duration known")
    return labels[:4] or ["no strong reason"]
