"""Tonepath command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise RuntimeError("Tonepath CLI dependencies are missing. Run `uv sync` first.") from exc

from tonepath.analysis import analyze_library
from tonepath.db import TonepathStore
from tonepath.doctor import run_doctor
from tonepath.enrichment import EnrichmentProvider, enrich_library
from tonepath.explanation import explain_candidate
from tonepath.models import CandidateScore
from tonepath.planner import plan_session
from tonepath.playback import MpvAdapter
from tonepath.playback_controller import PlaybackController
from tonepath.privacy import delete_profile, privacy_status
from tonepath.scanner import scan_directory
from tonepath.selector import select_path
from tonepath.tui import run_tui
from tonepath import config as tonepath_config


app = typer.Typer(help="Local-first music state-transition agent.")
config_app = typer.Typer(help="Manage local config.")
feedback_app = typer.Typer(help="Record local feedback.")
profile_app = typer.Typer(help="Inspect, export, or delete local profile data.")
privacy_app = typer.Typer(help="Inspect local privacy status.")
explain_app = typer.Typer(help="Explain selections.")

app.add_typer(config_app, name="config")
app.add_typer(feedback_app, name="feedback")
app.add_typer(profile_app, name="profile")
app.add_typer(privacy_app, name="privacy")
app.add_typer(explain_app, name="explain")

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
def scan(path: Annotated[Path | None, typer.Argument(help="Optional local music directory to scan.")] = None) -> None:
    """Scan local music directories into the Tonepath library."""

    paths = resolve_scan_paths(path)
    store = TonepathStore()
    total = 0
    scanned_dirs = 0
    skipped = 0
    try:
        for music_dir in paths:
            try:
                tracks = scan_directory(music_dir)
            except (FileNotFoundError, NotADirectoryError) as exc:
                skipped += 1
                console.print(f"Skipping {music_dir}: {exc}")
                continue
            for track in tracks:
                store.upsert_track(track)
            total += len(tracks)
            scanned_dirs += 1
    finally:
        store.close()

    console.print(f"Scanned {total} track(s) from {scanned_dirs} director(y/ies).")
    if skipped and scanned_dirs == 0:
        raise typer.Exit(code=1)


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
        session_id = store.save_session(plan)
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
            console.print(" ".join(command))
            console.print(f"Session {session_id} planned.")
            return

        controller = PlaybackController(store, adapter=adapter)
        process = controller.start(paths)
        console.print(f"Session {session_id} started with mpv PID {process.pid}.")
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


@app.command()
def analyze(features: Annotated[str, typer.Option(help="Feature tier: basic or vocalness.")] = "basic") -> None:
    """Run local audio feature analysis for scanned tracks."""

    if features not in {"basic", "vocalness"}:
        raise typer.BadParameter("only basic and vocalness feature analysis are implemented")
    store = TonepathStore()
    try:
        analyzed, skipped = analyze_library(store, features=features)
    finally:
        store.close()
    console.print(f"Analyzed {analyzed} track(s); skipped {skipped} missing track(s).")


@app.command()
def doctor() -> None:
    """Check local Tonepath dependencies."""

    console.print(run_doctor())


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
    """Inspect local profile data counts."""

    store = TonepathStore()
    summary = store.profile_summary()
    if json_output:
        console.print(summary)
        return
    table = Table("Table", "Rows")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)


@profile_app.command("export")
def profile_export() -> None:
    """Print a profile summary for export."""

    store = TonepathStore()
    console.print(store.profile_summary())


@profile_app.command("delete")
def profile_delete(all_data: Annotated[bool, typer.Option("--all", help="Delete all profile/session/feedback data.")] = False) -> None:
    """Delete local profile/session/feedback data."""

    if not all_data:
        console.print("Pass --all to confirm profile deletion.")
        raise typer.Exit(code=1)
    store = TonepathStore()
    delete_profile(store)
    console.print("Deleted local profile, feedback, play, and session data.")


@privacy_app.command("status")
def privacy_status_command() -> None:
    """Show local privacy status."""

    store = TonepathStore()
    console.print(privacy_status(store))


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
    store.record_feedback(feedback_type, session_id=store.current_session_id())
    console.print(f"Recorded feedback: {feedback_type}")


def resolve_scan_paths(path: Path | None) -> tuple[Path, ...]:
    """Return explicit scan path or configured music directories."""

    if path is not None:
        return (path.expanduser(),)
    return tonepath_config.load_config().expanded_music_dirs()


def render_plan(candidates: list[CandidateScore]) -> None:
    """Render selected candidates as a table."""

    table = Table("Phase", "Track", "Artist", "Confidence", "Score")
    for candidate in candidates:
        table.add_row(
            candidate.phase.label,
            candidate.track.title or "unknown",
            candidate.track.artist or "unknown",
            candidate.confidence,
            f"{candidate.score:.2f}",
        )
    console.print(table)
