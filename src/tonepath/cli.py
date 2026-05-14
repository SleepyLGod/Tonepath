"""Tonepath command-line interface."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

try:
    import typer
    from rich import box
    from rich.console import Console
    from rich.table import Table
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise RuntimeError("Tonepath CLI dependencies are missing. Run `uv sync` first.") from exc

from tonepath.analysis import AnalysisProgress, analyze_library
from tonepath.db import TonepathStore
from tonepath.doctor import run_doctor
from tonepath.enrichment import EnrichmentProvider, enrich_library
from tonepath.evaluation import evaluate_selection
from tonepath.explanation import explain_candidate
from tonepath.llm import llm_doctor, parse_prompt_with_llm
from tonepath.model_runtime import isolation_report, model_runtime_report, setup_essentia_tf_runtime
from tonepath.models import CandidateScore, Track
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
eval_app = typer.Typer(help="Evaluate local selection quality.")
models_app = typer.Typer(help="Manage local model runtimes.")
models_setup_app = typer.Typer(help="Set up local model runtimes.")
llm_app = typer.Typer(help="Inspect optional LLM integrations.")

app.add_typer(config_app, name="config")
app.add_typer(feedback_app, name="feedback")
app.add_typer(profile_app, name="profile")
app.add_typer(privacy_app, name="privacy")
app.add_typer(explain_app, name="explain")
app.add_typer(eval_app, name="eval")
app.add_typer(models_app, name="models")
models_app.add_typer(models_setup_app, name="setup")
app.add_typer(llm_app, name="llm")

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
def analyze(
    features: Annotated[str, typer.Option(help="Feature tier: basic, vocalness, mir, or tags.")] = "basic",
    method: Annotated[str, typer.Option(help="Analysis method: spectral, audio-separator, demucs-cli, essentia, or essentia-tf.")] = "spectral",
    only_missing: Annotated[bool, typer.Option("--only-missing", help="Analyze only tracks missing the requested feature.")] = False,
    changed_only: Annotated[bool, typer.Option("--changed-only", help="Analyze only files changed since the last scan.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-analyze tracks even when existing results are present.")] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Maximum number of eligible tracks to analyze.")] = None,
) -> None:
    """Run local audio feature analysis for scanned tracks."""

    if features not in {"basic", "vocalness", "mir", "tags"}:
        raise typer.BadParameter("only basic, vocalness, mir, and tags feature analysis are implemented")
    if features == "vocalness" and method not in {"spectral", "audio-separator", "demucs-cli"}:
        raise typer.BadParameter("only spectral, audio-separator, and demucs-cli vocalness methods are implemented")
    if features == "mir" and method != "essentia":
        raise typer.BadParameter("only essentia is supported for mir analysis")
    if features == "tags" and method not in {"essentia", "essentia-tf"}:
        raise typer.BadParameter("only essentia and essentia-tf are supported for tags analysis")
    if features == "basic" and method != "spectral":
        raise typer.BadParameter("--method is only supported with --features vocalness, mir, or tags")
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
    runtime = 0.0 if event.runtime_sec is None else event.runtime_sec
    console.print(
        f"result: energy={energy} bpm={bpm} vocalness={vocalness} source={event.result.feature_source} "
        f"confidence={event.result.confidence} runtime={runtime:.1f}s"
    )


def display_track(track: Track) -> str:
    """Return a compact track label for terminal progress output."""

    title = track.title or track.path.stem
    artist = track.artist or "unknown"
    return f"{title} - {artist}"


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
            "constraints": ["avoid_vocals"] if plan.request.no_vocals else [],
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


@eval_app.command("selection")
def eval_selection(
    prompt: Annotated[str, typer.Argument(help="State transition prompt to evaluate.")],
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of candidates to print.")] = 8,
    json_output: Annotated[bool, typer.Option("--json", help="Print stable JSON for comparison.")] = False,
) -> None:
    """Evaluate selection candidates without playback or profile writes."""

    if limit <= 0:
        raise typer.BadParameter("--limit must be greater than zero")
    store = TonepathStore()
    try:
        payload = evaluate_selection(store, prompt, limit)
    finally:
        store.close()

    if json_output:
        console.print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    render_eval_table(payload)


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


def eval_track_label(track: dict[str, object]) -> str:
    """Return a compact track label for evaluation tables."""

    title = track.get("title") or "unknown"
    artist = track.get("artist") or "unknown"
    return f"{title} - {artist}"


def eval_cell(value: object) -> str:
    """Render missing evaluation values without inventing facts."""

    return "--" if value is None else str(value)


def eval_feature_summary(features: dict[str, object]) -> str:
    """Return compact feature details for selection review."""

    return (
        f"src={eval_cell(features['source'])}\n"
        f"energy={eval_cell(features['energy'])} loud={eval_cell(features['loudness'])}\n"
        f"bpm={eval_cell(features['bpm'])} vocal={eval_cell(features['vocalness'])}"
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
        elif "feedback" in reason:
            labels.append("feedback adjusted")
        elif "genre" in reason:
            labels.append("genre signal")
        elif "duration is known" in reason:
            labels.append("duration known")
    return labels[:4] or ["no strong reason"]
