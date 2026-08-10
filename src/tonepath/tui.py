"""Textual terminal interface for Tonepath."""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from functools import partial
from typing import Any

from rich.text import Text

from tonepath import config, preparation as tonepath_preparation
from tonepath.db import TonepathStore
from tonepath.display import display_artist, fallback_track_label
from tonepath.experience import smart_plan_session
from tonepath.evaluation import evaluate_rerank
from tonepath.llm import provider_config
from tonepath.memory import (
    add_memory_log,
    build_memory_evidence,
    consolidate_memory_with_llm,
    memory_profile_text,
    memory_suggestions_from_llm,
    save_consolidated_memory_profile,
    write_memory_evidence,
)
from tonepath.model_runtime import model_runtime_status
from tonepath.models import CandidateScore, FeedbackType, SessionPlan
from tonepath.playback_controller import PlaybackController, PlaybackState
from tonepath.preparation import PreparationEvent, PreparationOptions, PreparationResult
from tonepath.profile import apply_suggestion, apply_suggestion_group, list_pending_suggestions, pending_suggestion_groups, profile_learning_hint, save_suggestions
from tonepath.readiness import LibraryStatus, library_status, readiness_blocks_session, readiness_label, status_next_action
from tonepath.session import SessionRunner
from tonepath.tui_theme import PALETTE_BY_KEY, PALETTES, TonepathPalette, next_theme, normalize_theme

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.theme import Theme
    from textual.worker import Worker, WorkerState
    from textual.widgets import DataTable, Header, Input, RichLog, Static, TextArea
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise RuntimeError("Textual is not installed. Run `uv sync` before launching the TUI.") from exc

from tonepath.tui_history import HistoryLoadResult, HistoryRerunRequest, HistoryScreen
from tonepath.tui_privacy import (
    PrivacyDataDeleted,
    PrivacyScreen,
    category_delete_completed,
    database_records_cleared,
)
from tonepath.tui_setup import SetupOutcome, SetupScreen


PROMPT_PLACEHOLDER = "我现在很烦，想半小时后进入写代码状态，不要人声"
PLAYBACK_MODES = ("Manual", "Continue Path", "Repeat One", "Repeat Path")


@dataclass(frozen=True)
class MemoryWorkerResult:
    """Result payload returned from a background memory task."""

    kind: str
    status_message: str
    event_message: str


@dataclass(frozen=True)
class RequestWorkerResult:
    """Intent plan returned from a background Smart request."""

    prompt: str
    plan: SessionPlan
    intent_note: str | None
    history_source_session_id: int | None


@dataclass(frozen=True)
class PlaybackPollResult:
    """Live mpv state read for one playback generation."""

    generation: int
    state: PlaybackState


class TonepathApp(App[None]):
    """Terminal session screen for a local Tonepath listening path."""

    TITLE = "Tonepath"
    SUB_TITLE = "Local state-transition player"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    #status-bar {
        height: 1;
        padding: 0 2;
        text-style: bold;
        background: $surface;
        color: $primary;
    }

    #prompt-input {
        height: 3;
        margin: 0 1;
        background: $panel;
        border: round $primary;
        border-title-color: $primary;
        border-title-align: right;
        color: $foreground;
    }

    #prompt-input:focus {
        background: $surface;
        border: round $accent;
        border-title-color: $accent;
    }

    #timeline {
        height: 3;
        margin: 0 1;
        padding: 0 2;
        text-style: bold;
        background: $panel;
        border: round $primary;
        border-title-color: $primary;
        border-title-align: right;
        color: $primary;
    }

    #body {
        height: 1fr;
        padding: 0 1;
    }

    #left-pane {
        width: 60%;
        min-width: 50;
        padding-right: 1;
    }

    #right-pane {
        width: 40%;
        min-width: 28;
    }

    #now-playing,
    #why-panel,
    #memory-input,
    #memory-profile,
    #memory-suggestions {
        padding: 1 2;
        background: $panel;
        border: round $surface;
        color: $foreground;
    }

    #now-playing {
        height: 10;
        margin-bottom: 1;
        border-left: heavy $primary;
        border-title-color: $primary;
    }

    #queue {
        height: 1fr;
        background: $panel;
        border: round $surface;
        border-title-color: $primary;
        color: $foreground;
    }

    #queue .datatable--header {
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #queue .datatable--cursor {
        text-style: bold;
        color: $foreground;
        background: $primary;
    }

    #queue .datatable--odd-row {
        background: $panel;
    }

    #queue .datatable--even-row {
        background: $surface;
    }

    #why-panel,
    #memory-input,
    #memory-profile,
    #memory-suggestions {
        height: 1fr;
        border-title-color: $secondary;
    }

    #memory-input:focus {
        border: round $accent;
        border-title-color: $accent;
    }

    #event-log {
        height: 8;
        margin: 0 1;
        background: $surface;
        border: round $panel;
        border-title-color: $foreground;
        color: $foreground;
    }

    #command-bar {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $foreground;
    }
    """

    BINDINGS = [
        Binding("/", "focus_prompt", "Prompt", show=False),
        Binding("n", "new_prompt", "New", show=False),
        Binding("space", "play", "Play", key_display="Space"),
        Binding("p", "play", "Play", show=False),
        Binding("left", "seek_backward", "Back 10s", show=False, priority=True),
        Binding("right", "seek_forward", "Forward 10s", show=False, priority=True),
        Binding("up", "volume_up", "Volume up", show=False, priority=True),
        Binding("down", "volume_down", "Volume down", show=False, priority=True),
        Binding("x", "stop_playback", "Stop", show=False),
        Binding(">", "next_track", "Next", key_display=">"),
        Binding("<", "previous_track", "Prev", key_display="<"),
        Binding("s", "skip", "Skip"),
        Binding("l", "like", "Like"),
        Binding("m", "cycle_playback_mode", "Mode"),
        Binding("ctrl+l", "history", "History", key_display="Ctrl+L", show=False, priority=True),
        Binding("d", "privacy", "Data & Privacy", key_display="d", show=False),
        Binding("c", "setup", "Setup", key_display="c", show=False),
        Binding("ctrl+o", "toggle_memory", "Memory", key_display="Ctrl+O", show=False, priority=True),
        Binding("shift+m", "toggle_memory", "Memory", key_display="M", show=False, priority=True),
        Binding("ctrl+s", "save_memory", "Save memory", show=False, priority=True),
        Binding("ctrl+enter", "save_and_learn_memory", "Save + learn", show=False, priority=True),
        Binding("ctrl+p", "memory_profile", "Memory profile", show=False, priority=True),
        Binding("ctrl+g", "memory_suggestions", "Memory suggestions", show=False, priority=True),
        Binding("shift+p", "memory_profile", "Profile", key_display="P", show=False),
        Binding("shift+g", "memory_suggestions", "Suggestions", key_display="G", show=False),
        Binding("j", "next_memory_suggestion", "Next suggestion", show=False),
        Binding("k", "previous_memory_suggestion", "Previous suggestion", show=False),
        Binding("enter", "apply_memory_suggestion", "Apply suggestion", show=False),
        Binding("t", "cycle_theme", "Theme"),
        Binding("i", "ai_assist", "AI Assist", show=False),
        Binding("?", "toggle_help", "Help", key_display="?"),
        Binding("e", "toggle_events", "Events", show=False),
        Binding("v", "no_vocals", "No vocals", show=False),
        Binding("a", "codex_audit", "Audit", show=False),
        Binding("r", "codex_rerank", "Rerank", show=False),
        Binding("+", "too_loud", "Quieter", show=False),
        Binding("-", "too_slow", "More energy", show=False),
        Binding("w", "why", "Why", show=False),
        Binding("escape", "blur_prompt", "Done", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self, prompt: str | None = None) -> None:
        super().__init__()
        self.initial_prompt = prompt
        self.store: TonepathStore | None = None
        self.runner: SessionRunner | None = None
        self.playback: PlaybackController | None = None
        self.playback_status = "Ready"
        self.playback_timer: Any | None = None
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.playback_generation = 0
        self.playback_state_busy = False
        self.playback_poll_failures = 0
        self.model_runtime_ready = False
        self.library_status: LibraryStatus | None = None
        self.readiness = "Needs setup"
        self.readiness_action = "Run `uv run tonepath setup --preset private`."
        self.intent_note: str | None = None
        self.playback_mode = "Manual"
        self.events_expanded = False
        self.right_panel = "why"
        self.memory_draft = ""
        self.memory_busy = False
        self.memory_worker_kind: str | None = None
        self.memory_status_message = ""
        self.memory_suggestions: list[dict[str, object]] = []
        self.selected_memory_suggestion_index = 0
        self.request_busy = False
        self.request_status_message = ""
        self.setup_prepare_busy = False
        self.setup_prepare_status = ""
        self.auto_setup_needed = should_auto_open_setup()
        self.pulse_tick = 0
        self.theme_key = normalize_theme(config.load_config().ui.theme)
        self.palette = PALETTE_BY_KEY[self.theme_key]

    def compose(self) -> ComposeResult:
        """Compose the terminal product surface with Textual built-ins."""

        yield Header()
        yield Static("", id="status-bar")
        yield Input(placeholder=PROMPT_PLACEHOLDER, id="prompt-input")
        yield Static("", id="timeline")
        with Horizontal(id="body"):
            with Vertical(id="left-pane"):
                yield Static("", id="now-playing")
                yield DataTable(id="queue")
            with Vertical(id="right-pane"):
                yield Static("", id="why-panel")
                yield TextArea(
                    "",
                    id="memory-input",
                    soft_wrap=True,
                    show_line_numbers=False,
                    placeholder="Write a private note, rant, monologue, or listening context... / 可以写吐槽、独白、心情和偏好",
                )
                yield Static("", id="memory-profile")
                yield DataTable(id="memory-suggestions")
        yield RichLog(id="event-log", wrap=True, markup=False)
        yield Static("", id="command-bar")

    def on_mount(self) -> None:
        """Load local state and render the first session view."""

        self.store = TonepathStore()
        self.install_themes()
        self.theme = self.theme_key
        self.apply_panel_titles()
        table = self.query_one("#queue", DataTable)
        table.add_columns("#", "Phase", "Track", "Fit", "Energy", "Conf")
        suggestions = self.query_one("#memory-suggestions", DataTable)
        suggestions.add_columns("#", "Type", "Scope", "Confidence", "Rationale")
        self.show_right_panel()

        self.model_runtime_ready = model_runtime_status().ready
        self.refresh_readiness()
        if self.auto_setup_needed:
            self.show_empty_library()
            self.push_setup_screen(first_run=True)
            return
        if not self.store.list_tracks():
            self.show_empty_library()
            return

        self.playback = PlaybackController(self.store)
        if readiness_blocks_session(self.readiness):
            self.log_event(f"Library is not ready: {self.readiness}. {self.readiness_action}")
        elif not self.model_runtime_ready:
            self.log_event("Better vocalness is available after `uv run tonepath models setup essentia-tf`.")
        if self.initial_prompt is None:
            self.render_intake()
            self.log_event("Type a listening goal, then press Enter.")
            self.query_one("#prompt-input", Input).focus()
            return
        self.create_session(self.initial_prompt)

    def on_unmount(self) -> None:
        """Close local storage when the terminal app exits."""

        if self.playback is not None:
            self.playback.stop_current()
        if self.store is not None:
            self.store.close()

    def action_quit(self) -> None:
        """Stop local playback before exiting the TUI."""

        if self.playback is not None:
            self.playback.stop_current()
        self.exit()

    def action_blur_prompt(self) -> None:
        """Leave prompt editing without submitting or clearing the request."""

        prompt_input = self.query_one("#prompt-input", Input)
        if not bool(getattr(prompt_input, "has_focus", False)):
            return
        prompt_input.blur()
        self.query_one("#command-bar", Static).update(self.command_bar_renderable(prompt_focused=False))

    def action_skip(self) -> None:
        """Skip the current candidate and refresh upcoming recommendations."""

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            return
        was_playing = self.playback_status == "Playing"
        self.apply_feedback("skip")
        if was_playing:
            self.start_current_playback(mark_previous_skipped=True)
            return
        candidate = self.runner.current() if self.runner is not None else None
        if candidate is None:
            self.playback_status = "No tracks"
            self.log_event("Skipped. No next track available.")
        else:
            self.log_event(f"Skipped to: {candidate.track.title or candidate.track.path.name}")
        self.refresh_session_view()

    def action_next_track(self) -> None:
        """Move to the next candidate without recording negative feedback."""

        self.navigate_track(1)

    def action_previous_track(self) -> None:
        """Move to the previous candidate without recording feedback."""

        self.navigate_track(-1)

    def navigate_track(self, direction: int) -> None:
        """Move through the current queue without changing recommendation feedback."""

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            return
        was_playing = self.playback_status == "Playing"
        moved = self.runner.move_next() if direction > 0 else self.runner.move_previous()
        if not moved:
            self.log_event("No next track." if direction > 0 else "No previous track.")
            return
        if not was_playing:
            self.live_playback_state = PlaybackState(False, False, None, None, None)
            self.pulse_tick = 0
        candidate = self.runner.current()
        if candidate is None:
            self.playback_status = "No tracks"
            self.refresh_session_view()
            return
        label = fallback_track_label(candidate.track.title, candidate.track.path.name)
        self.log_event(f"Moved to {'next' if direction > 0 else 'previous'} track without feedback: {label}")
        if was_playing:
            self.start_current_playback()
            return
        self.refresh_session_view()

    def action_play(self) -> None:
        """Start, pause, or resume playback for the current candidate."""

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            self.query_one("#prompt-input", Input).focus()
            return
        if self.playback is not None and self.playback_status in {"Playing", "Paused"}:
            try:
                if self.playback_status == "Playing":
                    self.playback.pause()
                    paused = True
                else:
                    self.playback.resume()
                    paused = False
            except RuntimeError as exc:
                self.handle_playback_control_error(exc)
                return
            self.playback_status = "Paused" if paused else "Playing"
            self.live_playback_state = PlaybackState(
                True,
                paused,
                self.live_playback_state.position_sec,
                self.live_playback_state.duration_sec,
                self.live_playback_state.volume,
            )
            self.log_event("Paused playback." if paused else "Resumed playback.")
            self.refresh_session_view()
            return
        self.start_current_playback()

    def action_seek_backward(self) -> None:
        """Seek backward ten seconds when the player surface has focus."""

        if self.forward_directional_key("left"):
            return
        self.seek_playback(-10.0)

    def action_seek_forward(self) -> None:
        """Seek forward ten seconds when the player surface has focus."""

        if self.forward_directional_key("right"):
            return
        self.seek_playback(10.0)

    def seek_playback(self, seconds: float) -> None:
        """Seek managed playback without changing queue or feedback state."""

        if self.playback is None or self.playback_status not in {"Playing", "Paused"}:
            self.log_event("Start playback before seeking.")
            return
        try:
            self.playback.seek_relative(seconds)
            self.live_playback_state = self.playback.state()
        except RuntimeError as exc:
            self.handle_playback_control_error(exc)
            return
        direction = "forward" if seconds > 0 else "back"
        self.log_event(f"Sought {direction} {abs(int(seconds))} seconds.")
        self.refresh_session_view()

    def action_volume_up(self) -> None:
        """Raise managed playback volume by five percent."""

        if self.forward_directional_key("up"):
            return
        self.adjust_playback_volume(5.0)

    def action_volume_down(self) -> None:
        """Lower managed playback volume by five percent."""

        if self.forward_directional_key("down"):
            return
        self.adjust_playback_volume(-5.0)

    def adjust_playback_volume(self, delta: float) -> None:
        """Adjust managed playback volume without changing queue state."""

        if self.playback is None or self.playback_status not in {"Playing", "Paused"}:
            self.log_event("Start playback before changing volume.")
            return
        try:
            volume = self.playback.adjust_volume(delta)
        except RuntimeError as exc:
            self.handle_playback_control_error(exc)
            return
        self.live_playback_state = PlaybackState(
            True,
            self.playback_status == "Paused",
            self.live_playback_state.position_sec,
            self.live_playback_state.duration_sec,
            volume,
        )
        self.log_event(f"Volume: {volume:.0f}%.")
        self.refresh_session_view()

    def forward_directional_key(self, direction: str) -> bool:
        """Keep arrow-key behavior inside text editors and History."""

        focused = self.focused
        if isinstance(self.screen, HistoryScreen):
            if direction == "up":
                self.screen.action_previous_history()
            elif direction == "down":
                self.screen.action_next_history()
            return True
        if isinstance(self.screen, PrivacyScreen):
            if direction == "up":
                self.screen.action_previous_category()
            elif direction == "down":
                self.screen.action_next_category()
            return True
        if isinstance(self.screen, SetupScreen):
            if direction == "up" and not isinstance(focused, Input):
                self.screen.action_previous_option()
            elif direction == "down" and not isinstance(focused, Input):
                self.screen.action_next_option()
            elif isinstance(focused, Input):
                if direction == "left":
                    focused.action_cursor_left()
                elif direction == "right":
                    focused.action_cursor_right()
            return True
        if isinstance(focused, Input):
            if direction == "left":
                focused.action_cursor_left()
            elif direction == "right":
                focused.action_cursor_right()
            return True
        if isinstance(focused, TextArea):
            getattr(focused, f"action_cursor_{direction}")()
            return True
        return False

    def handle_playback_control_error(self, exc: RuntimeError) -> None:
        """Stop playback that can no longer be controlled and report why."""

        if self.playback is not None:
            self.playback.stop_current()
        self.playback_generation += 1
        self.playback_state_busy = False
        self.playback_poll_failures = 0
        self.playback_status = "Stopped"
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.log_event(f"Playback control failed; stopped mpv: {exc}")
        self.refresh_session_view()

    def action_stop_playback(self) -> None:
        """Stop active playback without exiting the TUI."""

        if self.playback is None:
            self.log_event("No playback controller.")
            return
        stopped = self.playback.stop_current()
        self.playback_generation += 1
        self.playback_state_busy = False
        self.playback_poll_failures = 0
        self.playback_status = "Stopped"
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.pulse_tick = 0
        self.log_event("Stopped playback." if stopped else "No active Tonepath playback.")
        self.refresh_session_view()

    def action_cycle_playback_mode(self) -> None:
        """Cycle between manual, path, and repeat playback modes."""

        index = PLAYBACK_MODES.index(self.playback_mode)
        self.playback_mode = PLAYBACK_MODES[(index + 1) % len(PLAYBACK_MODES)]
        self.log_event(f"Playback mode: {self.playback_mode}.")
        self.refresh_session_view()

    def action_cycle_theme(self) -> None:
        """Cycle the TUI visual theme and persist the explicit user choice."""

        self.theme_key = next_theme(self.theme_key)
        self.palette = PALETTE_BY_KEY[self.theme_key]
        self.theme = self.theme_key
        settings = config.load_config()
        config.write_config(
            config.TonepathConfig(
                music_dirs=settings.music_dirs,
                data_dir=settings.data_dir,
                player=settings.player,
                network_mode=settings.network_mode,
                privacy=settings.privacy,
                models=settings.models,
                experience=settings.experience,
                ui=config.UiConfig(theme=self.theme_key),
                llm=settings.llm,
            )
        )
        self.log_event(f"Theme: {self.palette.label}.")
        self.refresh_session_view()

    def action_toggle_help(self) -> None:
        """Toggle the right panel between explanation and key help."""

        if self.right_panel == "memory":
            self.sync_memory_draft()
        self.right_panel = "why" if self.right_panel == "help" else "help"
        self.log_event("Showing help panel." if self.right_panel == "help" else "Showing why panel.")
        self.refresh_session_view()

    def action_ai_assist(self) -> None:
        """Show local AI Assist status without changing config or calling a model."""

        if self.right_panel == "memory":
            self.sync_memory_draft()
        self.right_panel = "why" if self.right_panel == "ai_assist" else "ai_assist"
        self.log_event("Showing AI Assist status." if self.right_panel == "ai_assist" else "Showing why panel.")
        self.refresh_session_view()

    def action_toggle_memory(self) -> None:
        """Toggle the private memory notes panel without losing its draft."""

        if self.right_panel == "memory":
            self.sync_memory_draft()
            self.right_panel = "why"
        else:
            self.right_panel = "memory"
        self.refresh_session_view()
        if self.right_panel == "memory":
            self.query_one("#memory-input", TextArea).focus()
            self.query_one("#command-bar", Static).update(self.command_bar_renderable(prompt_focused=False))

    def action_history(self) -> None:
        """Open listening history without changing the active player state."""

        if isinstance(self.screen, HistoryScreen):
            self.screen.dismiss(None)
            return
        if self.store is None:
            self.log_event("Local store is unavailable.")
            return
        self.push_screen(HistoryScreen(self.store), self.on_history_loaded)

    def action_privacy(self) -> None:
        """Open the local Data & Privacy browser outside text input focus."""

        if isinstance(self.screen, PrivacyScreen):
            return
        self.push_screen(PrivacyScreen())

    def action_setup(self) -> None:
        """Open setup outside text input focus without changing the current path."""

        if isinstance(self.screen, SetupScreen):
            return
        if self.setup_prepare_busy:
            self.log_event("Library preparation is already running. Wait for it to finish before changing setup.")
            return
        self.push_setup_screen(first_run=not config.config_path().exists())

    def push_setup_screen(self, *, first_run: bool) -> None:
        """Open the guided setup screen with current non-secret runtime state."""

        if self.right_panel == "memory":
            self.sync_memory_draft()
        runtime = model_runtime_status()
        model_ready = bool(runtime.ready) and bool(getattr(runtime, "affect_ready", runtime.ready))
        self.push_screen(
            SetupScreen(
                config.load_config(),
                first_run=first_run,
                model_ready=model_ready,
            ),
            self.on_setup_complete,
        )

    def on_setup_complete(self, outcome: SetupOutcome | None) -> None:
        """Persist confirmed setup and optionally start background preparation."""

        if outcome is None:
            self.log_event("Setup closed without saving changes.")
            self.refresh_session_view()
            return
        try:
            config.write_config(outcome.settings)
        except OSError as exc:
            self.log_event(f"Could not save setup: {exc}")
            self.refresh_session_view()
            return
        self.log_event(f"Setup saved: {outcome.settings.experience.mode.title()} experience.")
        self.refresh_readiness()
        if not outcome.prepare_requested:
            if self.runner is None and self.library_count() == 0:
                self.show_empty_library()
            else:
                self.refresh_session_view()
            return
        self.start_setup_preparation(outcome)

    def start_setup_preparation(self, outcome: SetupOutcome) -> None:
        """Run one confirmed library preparation task outside the Textual event loop."""

        if self.setup_prepare_busy:
            self.log_event("Library preparation is already running.")
            return
        self.setup_prepare_busy = True
        self.setup_prepare_status = "Preparing library in background; playback and the current queue continue."
        self.log_event(self.setup_prepare_status)
        self.refresh_session_view()
        options = PreparationOptions(
            paths=outcome.settings.expanded_music_dirs(),
            mode=outcome.settings.models.mode,
            setup_models=outcome.setup_models,
        )
        self.run_worker(
            partial(self.setup_preparation_job, options),
            name="setup-prepare",
            group="setup-prepare",
            description=self.setup_prepare_status,
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def setup_preparation_job(self, options: PreparationOptions) -> PreparationResult:
        """Run shared preparation and forward stage messages to the TUI thread."""

        return tonepath_preparation.run_preparation(
            options,
            on_event=lambda event: self.call_from_thread(self.on_setup_preparation_event, event),
        )

    def on_setup_preparation_event(self, event: PreparationEvent) -> None:
        """Show one background preparation stage without modifying the current queue."""

        self.setup_prepare_status = event.message
        self.log_event(event.message)
        if self.runner is None:
            self.render_intake()
        else:
            self.query_one("#status-bar", Static).update(self.status_bar_text(extra="Preparing library"))

    def on_privacy_data_deleted(self, event: PrivacyDataDeleted) -> None:
        """Reconcile in-memory player state with components actually deleted."""

        result = event.result
        changed = set(result.changed_categories)
        memory_cleared = category_delete_completed(result, "memory")
        personalization_cleared = category_delete_completed(result, "personalization")
        if memory_cleared:
            self.memory_draft = ""
            self.memory_status_message = "Memory was deleted from Tonepath active storage."
            self.query_one("#memory-input", TextArea).load_text("")
            if self.right_panel in {"memory", "memory_profile"}:
                self.right_panel = "why"
        if personalization_cleared:
            self.memory_suggestions = []
            self.selected_memory_suggestion_index = 0
            if self.right_panel == "memory_suggestions":
                self.right_panel = "why"
        if database_records_cleared(result, "history"):
            if self.playback is not None:
                self.playback.stop_current()
            self.playback_generation += 1
            self.playback_state_busy = False
            self.playback_poll_failures = 0
            self.live_playback_state = PlaybackState(False, False, None, None, None)
            self.runner = None
            self.intent_note = None
            self.request_status_message = "Listening History was deleted. Enter a new Request to create a path."
            self.playback_status = "Ready"
            self.pulse_tick = 0
            self.right_panel = "why"
            prompt_input = self.query_one("#prompt-input", Input)
            prompt_input.value = ""
            prompt_input.blur()
            self.render_intake()
            self.log_event("Listening History deleted; playback stopped and the current queue was cleared.")
        elif changed or memory_cleared or personalization_cleared:
            affected = changed | {
                category
                for category, completed in (
                    ("memory", memory_cleared),
                    ("personalization", personalization_cleared),
                )
                if completed
            }
            self.refresh_right_panel()
            self.query_one("#command-bar", Static).update(self.command_bar_renderable())
            self.log_event(
                f"Privacy deletion updated {', '.join(sorted(affected))}. Current queue is unchanged."
            )

    def on_history_loaded(
        self,
        result: HistoryLoadResult | HistoryRerunRequest | None,
    ) -> None:
        """Handle an exact history load or a fresh run of its original request."""

        if result is None:
            return
        if isinstance(result, HistoryRerunRequest):
            self.start_request_planning(
                result.prompt,
                history_source_session_id=result.source_session_id,
            )
            return
        if self.playback is not None:
            self.playback.stop_current()
        self.runner = result.runner
        self.intent_note = None
        self.playback = self.playback or PlaybackController(self.store)
        self.playback_status = "Ready"
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.pulse_tick = 0
        self.right_panel = "why"
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = self.runner.active_plan().request.prompt
        prompt_input.blur()
        for item in result.omitted:
            label = item.title or item.path.name
            self.log_event(f"History omitted missing file: {label}")
        self.log_event(
            f"Loaded exact history path from session {result.source_session_id}. Press Space to play."
        )
        self.refresh_session_view()

    def action_save_memory(self) -> None:
        """Save the current memory draft to the local memory log only."""

        if not self.save_memory_draft():
            return
        self.memory_status_message = "Memory saved locally. Use Ctrl+Enter when you want to update the profile."
        self.right_panel = "memory"
        self.refresh_session_view()

    def action_save_and_learn_memory(self) -> None:
        """Save the current memory draft and consolidate new logs with explicit AI Assist."""

        if self.memory_busy:
            self.show_memory_task_busy("memory")
            return
        self.sync_memory_draft()
        had_draft = bool(self.memory_draft.strip())
        saved = self.save_memory_draft(allow_empty=True)
        if had_draft and not saved:
            return
        settings = config.load_config()
        if not self.llm_ready(settings):
            if saved:
                self.memory_status_message = "Memory saved locally. AI Assist is not ready, so profile was not updated."
            else:
                self.memory_status_message = "AI Assist is not ready. Run setup smart with send-to-llm, then try again."
            self.right_panel = "memory"
            self.log_event(self.memory_status_message)
            self.refresh_session_view()
            return
        if self.store is None:
            self.memory_status_message = "Local store is unavailable."
            self.right_panel = "memory"
            self.refresh_session_view()
            return
        self.start_memory_worker("consolidate", "memory", self.consolidate_memory_job)

    def action_memory_profile(self) -> None:
        """Show the consolidated memory profile without clearing the memory draft."""

        self.sync_memory_draft()
        self.right_panel = "memory_profile"
        self.refresh_session_view()

    def action_memory_suggestions(self) -> None:
        """Generate memory-derived profile suggestions without applying them."""

        if self.memory_busy:
            self.show_memory_task_busy("memory_suggestions")
            return
        self.sync_memory_draft()
        settings = config.load_config()
        if not self.llm_ready(settings):
            self.memory_status_message = "AI Assist is not ready. Run `uv run tonepath setup --preset smart --send-to-llm`."
            self.right_panel = "memory_suggestions"
            self.refresh_session_view()
            return
        if self.store is None:
            self.memory_status_message = "Local store is unavailable."
            self.right_panel = "memory_suggestions"
            self.refresh_session_view()
            return
        self.start_memory_worker("suggestions", "memory_suggestions", self.memory_suggestions_job)

    def start_memory_worker(self, kind: str, panel: str, work: Any) -> None:
        """Start one background memory task without blocking playback controls."""

        self.memory_busy = True
        self.memory_worker_kind = kind
        self.right_panel = panel
        self.memory_status_message = "Memory learning in background... playback continues."
        self.refresh_session_view()
        self.log_event(self.memory_status_message)
        self.run_worker(
            work,
            name=f"memory-{kind}",
            group="memory",
            description=self.memory_status_message,
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def show_memory_task_busy(self, panel: str) -> None:
        """Tell the user an existing memory task is still running."""

        self.right_panel = panel
        self.memory_status_message = "Memory task already running. Playback continues."
        self.refresh_session_view()

    def consolidate_memory_job(self) -> MemoryWorkerResult:
        """Build evidence and consolidate memory profile off the TUI thread."""

        store = TonepathStore()
        try:
            evidence = build_memory_evidence(store)
            new_logs = evidence.get("new_memory_logs", [])
            if not isinstance(new_logs, list) or not new_logs:
                return MemoryWorkerResult(
                    kind="consolidate",
                    status_message="No new memory logs to consolidate.",
                    event_message="No new memory logs to consolidate. Current queue is unchanged.",
                )
            evidence_path = write_memory_evidence(evidence)
            profile_markdown = consolidate_memory_with_llm(evidence)
            save_consolidated_memory_profile(store, evidence, profile_markdown)
            return MemoryWorkerResult(
                kind="consolidate",
                status_message="Memory profile updated. Suggestions are still advisory until applied.",
                event_message=f"Memory profile updated from {evidence_path}. Current queue is unchanged.",
            )
        except RuntimeError as exc:
            message = f"Memory saved, but profile update failed: {exc}"
            return MemoryWorkerResult(kind="consolidate", status_message=message, event_message=message)
        finally:
            store.close()

    def memory_suggestions_job(self) -> MemoryWorkerResult:
        """Generate pending memory suggestions off the TUI thread."""

        store = TonepathStore()
        try:
            evidence = build_memory_evidence(store)
            evidence_path = write_memory_evidence(evidence)
            suggestions = memory_suggestions_from_llm(evidence)
            save_suggestions(evidence, suggestions, source="memory-llm")
            if suggestions:
                status = f"Generated {len(suggestions)} pending memory suggestion(s)."
            else:
                status = "No memory suggestions yet; more evidence is needed."
            return MemoryWorkerResult(
                kind="suggestions",
                status_message=status,
                event_message=f"Memory suggestions checked: {evidence_path}. Current queue is unchanged.",
            )
        except RuntimeError as exc:
            message = f"Memory suggestions failed: {exc}"
            return MemoryWorkerResult(kind="suggestions", status_message=message, event_message=message)
        finally:
            store.close()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Apply completed request or memory work on the TUI thread."""

        worker = event.worker
        terminal_states = {WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED}
        if event.state not in terminal_states:
            return
        if worker.group == "request":
            self.finish_request_worker(worker, event.state)
            return
        if worker.group == "playback-state":
            self.finish_playback_state_worker(worker, event.state)
            return
        if worker.group == "setup-prepare":
            self.finish_setup_preparation_worker(worker, event.state)
            return
        if worker.group != "memory":
            return
        self.memory_busy = False
        self.memory_worker_kind = None
        if event.state == WorkerState.SUCCESS:
            result = worker.result
            if isinstance(result, MemoryWorkerResult):
                self.memory_status_message = result.status_message
                if result.kind == "suggestions":
                    self.load_memory_suggestions()
                self.log_event(result.event_message)
            else:
                self.memory_status_message = "Memory task finished, but returned an unreadable result."
                self.log_event(self.memory_status_message)
        elif event.state == WorkerState.ERROR:
            self.memory_status_message = f"Memory task failed: {worker.error}. Try the CLI memory command as a fallback."
            self.log_event(self.memory_status_message)
        else:
            self.memory_status_message = "Memory task was cancelled. Current queue is unchanged."
            self.log_event(self.memory_status_message)
        self.refresh_session_view()

    def finish_setup_preparation_worker(self, worker: Worker[Any], state: WorkerState) -> None:
        """Apply completed preparation without replacing the active path."""

        self.setup_prepare_busy = False
        if state == WorkerState.SUCCESS:
            if isinstance(worker.result, PreparationResult):
                result = worker.result
                self.model_runtime_ready = result.runtime_ready
                self.library_status = result.status
                if self.runner is None and result.status.tracks > 0:
                    self.playback_status = "Ready"
                settings = config.load_config()
                self.readiness = readiness_label(result.status, result.runtime_ready, settings)
                self.readiness_action = status_next_action(result.status, result.runtime_ready, settings)
                failure_note = f" {len(result.failures)} file(s) need review." if result.failures else ""
                self.setup_prepare_status = f"Library preparation finished: {self.readiness}.{failure_note}"
            else:
                self.setup_prepare_status = (
                    "Library preparation returned an unreadable result. Retry with `uv run tonepath prepare`."
                )
        elif state == WorkerState.ERROR:
            self.setup_prepare_status = (
                f"Library preparation failed: {worker.error}. Retry with `uv run tonepath prepare`."
            )
        else:
            self.setup_prepare_status = "Library preparation was cancelled. Retry with `uv run tonepath prepare`."
        self.log_event(self.setup_prepare_status)
        if self.runner is None and self.library_count() == 0:
            self.show_empty_library()
        else:
            self.refresh_session_view()

    def action_next_memory_suggestion(self) -> None:
        """Move down in the memory suggestion list."""

        if self.right_panel != "memory_suggestions":
            return
        items = self.memory_suggestion_items()
        if not items:
            return
        self.selected_memory_suggestion_index = min(self.selected_memory_suggestion_index + 1, len(items) - 1)
        self.refresh_memory_suggestions()

    def action_previous_memory_suggestion(self) -> None:
        """Move up in the memory suggestion list."""

        if self.right_panel != "memory_suggestions":
            return
        items = self.memory_suggestion_items()
        if not items:
            return
        self.selected_memory_suggestion_index = max(self.selected_memory_suggestion_index - 1, 0)
        self.refresh_memory_suggestions()

    def action_apply_memory_suggestion(self) -> None:
        """Apply the selected memory-derived suggestion for future requests only."""

        if self.right_panel != "memory_suggestions" or self.store is None:
            return
        items = self.memory_suggestion_items()
        if not items:
            self.memory_status_message = "No pending memory suggestions to apply."
            self.refresh_memory_suggestions()
            return
        index = min(self.selected_memory_suggestion_index, len(items) - 1)
        item = items[index]
        try:
            if item["kind"] == "group":
                result = apply_suggestion_group(self.store, str(item["id"]))
                applied = ", ".join(str(value) for value in result.get("applied", [])) or "none"
                skipped = ", ".join(str(value) for value in result.get("skipped", [])) or "none"
                self.memory_status_message = f"Applied group {item['id']} for future requests. Applied: {applied}; skipped: {skipped}."
            else:
                rule = apply_suggestion(self.store, str(item["id"]))
                self.memory_status_message = f"Applied {rule.rule_type} for future requests."
        except RuntimeError as exc:
            self.memory_status_message = str(exc)
        self.load_memory_suggestions()
        self.log_event(f"{self.memory_status_message} Rerun Request to use it.")
        self.refresh_session_view()

    def action_toggle_events(self) -> None:
        """Expand or collapse the event log panel."""

        self.events_expanded = not self.events_expanded
        log = self.query_one("#event-log", RichLog)
        log.styles.height = 16 if self.events_expanded else 8
        self.apply_panel_titles()

    def action_focus_prompt(self) -> None:
        """Focus the prompt input bar."""

        self.query_one("#prompt-input", Input).focus()
        self.query_one("#command-bar", Static).update(self.command_bar_renderable(prompt_focused=True))

    def action_new_prompt(self) -> None:
        """Return to the prompt intake state."""

        if self.playback is not None:
            self.playback.stop_current()
        self.runner = None
        self.playback_status = "Ready"
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.pulse_tick = 0
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = ""
        self.render_intake()
        prompt_input.focus()
        self.log_event("New request. Type a listening goal, then press Enter.")

    def action_like(self) -> None:
        """Record local like feedback."""

        self.apply_feedback("like")

    def action_no_vocals(self) -> None:
        """Apply a no-vocals constraint to upcoming recommendations."""

        self.apply_feedback("no-vocals")

    def action_too_loud(self) -> None:
        """Reduce upcoming energy targets."""

        self.apply_feedback("too-loud")

    def action_too_slow(self) -> None:
        """Raise upcoming energy targets."""

        self.apply_feedback("too-slow")

    def action_why(self) -> None:
        """Write the current auditable explanation to the event log."""

        if self.right_panel == "memory":
            self.sync_memory_draft()
        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            return
        self.log_event(self.runner.current_explanation())

    def action_codex_audit(self) -> None:
        """Show the explicit Codex audit command for the active session."""

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            return
        prompt = self.runner.active_plan().request.prompt
        self.log_event(f"Codex audit: uv run tonepath eval audit {prompt!r} --codex --web --limit 12")

    def action_codex_rerank(self) -> None:
        """Show how to get a Codex rerank recommendation."""

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            return
        preview = self.latest_codex_preview()
        if preview is None:
            self.log_event("Run Codex audit first; use demote/reject decisions as rerank guidance.")
            return
        for line in preview:
            self.log_event(line)

    def apply_feedback(self, feedback_type: FeedbackType) -> None:
        """Apply one feedback action to the active session."""

        if self.runner is None:
            self.log_event("No active session.")
            return
        message = self.runner.apply_feedback(feedback_type)
        self.log_event(message)
        self.log_event(profile_learning_hint())
        self.refresh_session_view()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Create a new local session when the prompt input is submitted."""

        if event.input.id != "prompt-input":
            return
        self.start_request_planning(event.value)

    def start_request_planning(
        self,
        prompt: str,
        *,
        history_source_session_id: int | None = None,
    ) -> None:
        """Plan a Smart request off-thread while keeping the current path active."""

        if self.request_busy:
            self.request_status_message = "Request planning already running. Current playback continues."
            self.log_event(self.request_status_message)
            self.refresh_session_view()
            return
        cleaned = prompt.strip()
        if not cleaned or self.store is None:
            self.create_session(
                prompt,
                history_source_session_id=history_source_session_id,
            )
            return
        self.refresh_readiness()
        if readiness_blocks_session(self.readiness):
            self.create_session(
                cleaned,
                history_source_session_id=history_source_session_id,
            )
            return
        settings = config.load_config()
        if not self.llm_ready(settings):
            self.create_session(
                cleaned,
                history_source_session_id=history_source_session_id,
            )
            return
        self.request_busy = True
        self.request_status_message = "Planning next path in background; current playback continues."
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = cleaned
        prompt_input.blur()
        self.log_event(self.request_status_message)
        self.refresh_session_view()
        self.run_worker(
            partial(
                self.request_planning_job,
                cleaned,
                settings,
                history_source_session_id,
            ),
            name="request-plan",
            group="request",
            description=self.request_status_message,
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def request_planning_job(
        self,
        prompt: str,
        settings: config.TonepathConfig,
        history_source_session_id: int | None,
    ) -> RequestWorkerResult:
        """Parse one Smart request without touching TUI or SQLite state."""

        plan, note = smart_plan_session(prompt, settings)
        return RequestWorkerResult(
            prompt=prompt,
            plan=plan,
            intent_note=note,
            history_source_session_id=history_source_session_id,
        )

    def finish_request_worker(self, worker: Worker[Any], state: WorkerState) -> None:
        """Finish background request planning without replacing state on failure."""

        self.request_busy = False
        if state == WorkerState.SUCCESS and isinstance(worker.result, RequestWorkerResult):
            self.activate_session(worker.result)
            return
        if state == WorkerState.ERROR:
            self.request_status_message = f"Request planning failed: {worker.error}. Current path is unchanged."
        elif state == WorkerState.CANCELLED:
            self.request_status_message = "Request planning was cancelled. Current path is unchanged."
        else:
            self.request_status_message = "Request planning returned an unreadable result. Current path is unchanged."
        self.log_event(self.request_status_message)
        self.refresh_session_view()

    def create_session(
        self,
        prompt: str,
        *,
        history_source_session_id: int | None = None,
    ) -> None:
        """Create a local session from a user prompt and refresh the TUI."""

        cleaned = prompt.strip()
        if not cleaned:
            self.log_event("Prompt is empty. Type a listening goal first.")
            self.query_one("#prompt-input", Input).focus()
            return
        if self.store is None:
            self.log_event("Local store is unavailable.")
            return
        self.refresh_readiness()
        if readiness_blocks_session(self.readiness):
            self.runner = None
            self.playback_status = "Needs setup"
            self.render_intake()
            self.log_event(f"Not ready for recommendations: {self.readiness}.")
            self.log_event(self.readiness_action)
            self.query_one("#prompt-input", Input).focus()
            return
        settings = config.load_config()
        plan, note = smart_plan_session(cleaned, settings)
        self.activate_session(
            RequestWorkerResult(
                prompt=cleaned,
                plan=plan,
                intent_note=note,
                history_source_session_id=history_source_session_id,
            )
        )

    def activate_session(self, result: RequestWorkerResult) -> None:
        """Persist and activate a completed request plan without autoplay."""

        if self.store is None:
            self.request_status_message = "Local store is unavailable. Current path is unchanged."
            self.log_event(self.request_status_message)
            self.refresh_session_view()
            return
        try:
            new_runner = SessionRunner(self.store, result.prompt, plan=result.plan)
        except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
            self.request_status_message = (
                f"Could not activate planned path: {exc}. Current path is unchanged."
            )
            self.log_event(self.request_status_message)
            self.refresh_session_view()
            return
        if self.playback is not None:
            self.playback.stop_current()
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.pulse_tick = 0
        self.right_panel = "why"
        self.intent_note = result.intent_note
        self.runner = new_runner
        self.playback = self.playback or PlaybackController(self.store)
        self.playback_status = "Ready"
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = result.prompt
        prompt_input.blur()
        if result.intent_note:
            self.log_event(result.intent_note)
        if result.history_source_session_id is None:
            self.request_status_message = f"Ready. Press Space to play. Session: {result.prompt}"
        else:
            self.request_status_message = (
                f"Reran Request from session {result.history_source_session_id} with current "
                "recommendation logic. Press Space to play."
            )
        self.log_event(self.request_status_message)
        self.refresh_session_view()

    def start_current_playback(self, mark_previous_skipped: bool = False) -> None:
        """Start playback for the current candidate if one exists."""

        if self.runner is None or self.playback is None:
            return
        candidate = self.runner.current()
        if candidate is None:
            self.playback_status = "No tracks"
            self.log_event("No current track to play.")
            self.refresh_session_view()
            return
        try:
            self.playback.replace(
                [candidate.track.path],
                session_id=self.runner.session_id,
                track_id=candidate.track.id,
                mark_current_skipped=mark_previous_skipped,
            )
        except RuntimeError as exc:
            self.log_event(str(exc))
            return
        self.playback_generation += 1
        self.playback_poll_failures = 0
        self.playback_status = "Playing"
        try:
            self.live_playback_state = self.playback.state()
        except RuntimeError as exc:
            self.handle_playback_control_error(exc)
            return
        self.pulse_tick = 0
        self.log_event(f"Playing: {fallback_track_label(candidate.track.title, candidate.track.path.name)}")
        self.ensure_playback_polling()
        self.refresh_session_view()

    def ensure_playback_polling(self) -> None:
        """Start a lightweight TUI-local poller for natural mpv exits."""

        if self.playback_timer is None:
            self.playback_timer = self.set_interval(0.5, self.poll_playback_finished)

    def poll_playback_finished(self) -> None:
        """Update local session state when mpv exits without an explicit stop."""

        if self.playback is None:
            return
        if not self.playback.finish_if_exited():
            if (
                self.playback_status in {"Playing", "Paused"}
                and self.runner is not None
                and not self.playback_state_busy
            ):
                self.playback_state_busy = True
                generation = self.playback_generation
                self.run_worker(
                    partial(self.read_playback_state_job, generation),
                    name="playback-state",
                    group="playback-state",
                    description="Reading mpv playback state",
                    exit_on_error=False,
                    exclusive=True,
                    thread=True,
                )
            return
        if self.runner is not None and self.playback_mode == "Repeat One":
            self.log_event("Repeating current track.")
            self.start_current_playback()
            return
        if self.runner is not None and self.playback_mode in {"Continue Path", "Repeat Path"}:
            if self.runner.move_next():
                self.log_event("Continuing to next track.")
                self.start_current_playback()
                return
            if self.playback_mode == "Repeat Path" and self.runner.move_to_start():
                self.log_event("Repeating path from the first track.")
                self.start_current_playback()
                return
        self.playback_status = "Finished"
        self.live_playback_state = PlaybackState(False, False, None, None, None)
        self.pulse_tick = 0
        self.log_event("Playback finished.")
        self.refresh_session_view()

    def read_playback_state_job(self, generation: int) -> PlaybackPollResult:
        """Read live mpv state without blocking the Textual event loop."""

        if self.playback is None:
            raise RuntimeError("No playback controller is active.")
        return PlaybackPollResult(generation=generation, state=self.playback.state())

    def finish_playback_state_worker(self, worker: Worker[Any], state: WorkerState) -> None:
        """Apply one background telemetry result without overreacting to a transient timeout."""

        self.playback_state_busy = False
        if self.playback_status not in {"Playing", "Paused"}:
            return
        if state == WorkerState.SUCCESS and isinstance(worker.result, PlaybackPollResult):
            result = worker.result
            if result.generation != self.playback_generation:
                return
            self.playback_poll_failures = 0
            self.live_playback_state = result.state
            self.playback_status = "Paused" if result.state.paused else "Playing"
            if self.playback_status == "Playing":
                self.pulse_tick += 1
            self.refresh_playback_surfaces()
            return
        if state == WorkerState.ERROR:
            self.playback_poll_failures += 1
            if self.playback_poll_failures >= 3:
                error = worker.error
                detail = error if isinstance(error, RuntimeError) else RuntimeError(str(error))
                self.handle_playback_control_error(detail)

    def refresh_playback_surfaces(self) -> None:
        """Refresh only widgets driven by live playback telemetry."""

        self.query_one("#now-playing", Static).update(self.now_playing_renderable())
        self.query_one("#status-bar", Static).update(self.status_bar_text())
        self.query_one("#command-bar", Static).update(self.command_bar_renderable())

    def refresh_session_view(self) -> None:
        """Refresh timeline, queue, why panel, and playback status."""

        if self.runner is None:
            self.render_intake()
            return

        self.apply_panel_titles()
        if self.setup_prepare_busy:
            planning = "Preparing library"
        else:
            planning = "Planning next path" if self.request_busy else None
        self.query_one("#status-bar", Static).update(self.status_bar_text(extra=planning))
        self.query_one("#timeline", Static).update(self.timeline_renderable())
        self.query_one("#now-playing", Static).update(self.now_playing_renderable())
        self.refresh_right_panel()
        self.query_one("#command-bar", Static).update(self.command_bar_renderable())
        self.refresh_queue()

    def refresh_queue(self) -> None:
        """Refresh the queue table."""

        if self.runner is None:
            return
        table = self.query_one("#queue", DataTable)
        table.clear(columns=False)
        candidates = []
        current = self.runner.current()
        if current is not None:
            candidates.append(("now", current))
        candidates.extend((f"+{index}", candidate) for index, candidate in enumerate(self.runner.upcoming(), start=1))

        for position, candidate in candidates:
            current = position == "now"
            table.add_row(
                queue_cell(queue_marker(position), current=current, align="center", palette=self.palette),
                queue_cell(candidate.phase.label, current=current, palette=self.palette),
                queue_cell(truncate(fallback_track_label(candidate.track.title, candidate.track.path.name), 28), current=current, palette=self.palette),
                fit_cell(self.fit_label(candidate), current=current, palette=self.palette),
                queue_cell(self.energy_text(candidate.track.id), current=current, palette=self.palette),
                queue_cell(confidence_label(candidate.confidence), current=current, palette=self.palette),
            )

    def timeline_text(self) -> str:
        """Return a compact path timeline for the current plan."""

        if self.runner is None:
            return "Tonepath"
        request = self.runner.active_plan().request
        labels = [phase.label for phase in self.runner.active_plan().phases]
        if labels and labels[-1] == request.target_state:
            labels = labels[:-1]
        path = "  ◇  ".join([request.source_state, *labels, request.target_state])
        return f"{path} · {request.duration_sec // 60}m"

    def timeline_renderable(self) -> Text:
        """Return a styled path timeline with subtle phase color transitions."""

        if self.runner is None:
            return Text("Tonepath", style=f"bold {self.palette.primary}")
        request = self.runner.active_plan().request
        labels = [phase.label for phase in self.runner.active_plan().phases]
        if labels and labels[-1] == request.target_state:
            labels = labels[:-1]
        parts = [request.source_state, *labels, request.target_state]
        text = Text()
        styles = [self.palette.warning, self.palette.primary, self.palette.secondary, self.palette.success]
        for index, part in enumerate(parts):
            if index:
                text.append("  ◇  ", style=self.palette.muted)
            text.append(part, style=f"bold {styles[min(index, len(styles) - 1)]}")
        text.append(f" · {request.duration_sec // 60}m", style=f"bold {self.palette.primary}")
        return text

    def now_playing_text(self) -> str:
        """Return the now-playing panel text."""

        if self.runner is None:
            return "No active session."
        candidate = self.runner.current()
        if candidate is None:
            return "Queue is empty. Run `tonepath scan` to add local music."
        features = self.store.get_features(candidate.track.id) if self.store is not None and candidate.track.id else None
        energy = "--" if features is None or features.energy is None else f"{features.energy:.2f}"
        loudness = "--" if features is None or features.loudness is None else f"{features.loudness:.1f} dBFS"
        bpm = bpm_text(features.bpm if features is not None else None)
        return "\n".join(
            [
                f"{self.playback_status} · {candidate.phase.label} · {self.playback_mode} · {confidence_label(candidate.confidence)}",
                truncate(fallback_track_label(candidate.track.title, candidate.track.path.name), 38),
                truncate(display_artist(candidate.track), 38),
                (
                    f"{self.pulse_text(features.energy if features is not None else None, features.arousal_estimate if features is not None else None)}"
                    f" · E {energy} · {bpm} BPM · {loudness}"
                ),
                f"{self.progress_text()} · vol {self.volume_text()}",
            ]
        )

    def now_playing_renderable(self) -> Text:
        """Return styled now-playing content."""

        lines = self.now_playing_text().splitlines()
        text = Text()
        if not lines:
            return text
        text.append(playback_symbol(self.playback_status), style=f"bold {self.palette.primary}")
        text.append(" ", style=self.palette.primary)
        text.append(lines[0], style=f"bold {self.palette.primary}")
        for index, line in enumerate(lines[1:], start=1):
            text.append("\n")
            if index == 1:
                text.append(line, style=f"bold {self.palette.text}")
            elif index == 2:
                text.append(line, style=self.palette.muted)
            elif "▮" in line or "▯" in line or "━" in line:
                text.append(line, style=self.palette.primary)
            else:
                text.append(line, style=self.palette.muted)
        return text

    def progress_text(self) -> str:
        """Return the progress reported by the managed mpv process."""

        duration = self.live_playback_state.duration_sec
        elapsed = self.live_playback_state.position_sec
        if duration is None or duration <= 0 or elapsed is None:
            return "--:-- ──────────── --:--"
        elapsed = min(max(elapsed, 0.0), duration)
        return f"{format_clock(elapsed)} {progress_bar(elapsed, duration)} {format_clock(duration)}"

    def volume_text(self) -> str:
        """Return the volume reported by the managed mpv process."""

        volume = self.live_playback_state.volume
        return "--" if volume is None else f"{volume:.0f}%"

    def pulse_text(self, energy: float | None, arousal: float | None = None) -> str:
        """Return a decorative energy pulse, not a real-time audio spectrum."""

        base = energy if energy is not None else arousal
        if base is None:
            return "▁▁▁▁▁▁▁▁"
        level = min(max(base, 0.0), 1.0)
        return pulse_meter(level, self.pulse_tick if self.playback_status == "Playing" else 0)

    def why_panel_text(self) -> str:
        """Return a compact explanation preview for the right panel."""

        if self.right_panel == "help":
            return self.help_panel_text()
        if self.right_panel == "ai_assist":
            return self.ai_assist_panel_text()
        if self.runner is None:
            return "Why panel\n\nA verifiable explanation appears after Tonepath creates a listening path."
        candidate = self.runner.current()
        if candidate is None:
            return "Why panel\n\nNo current track."
        features = self.store.get_features(candidate.track.id) if self.store is not None and candidate.track.id else None
        energy = "unknown" if features is None or features.energy is None else f"{features.energy:.2f}"
        loudness = "unknown" if features is None or features.loudness is None else f"{features.loudness:.1f} dBFS"
        bpm = bpm_text(features.bpm if features is not None else None)
        vocalness = vocalness_text(features.vocalness if features is not None else None)
        unknowns = []
        if features is None or features.bpm is None:
            unknowns.append("BPM")
        if features is None or features.vocalness is None:
            unknowns.append("vocalness")
        unknown = "none" if not unknowns else " · ".join(unknowns)
        return "\n".join(
            [
                "Why",
                self.human_fit_text(candidate),
                "Evidence",
                f"Confidence {confidence_label(candidate.confidence)} · Energy {energy}",
                f"BPM {bpm} · Loudness {loudness}",
                f"Vocalness {vocalness}",
                f"Missing evidence: {unknown}",
            ]
        )

    def human_fit_text(self, candidate: CandidateScore) -> str:
        """Return a short user-facing explanation for the current candidate."""

        notes: list[str] = []
        if any("semantic risk" in reason for reason in candidate.reasons):
            notes.append("has a calm-fit caution from local tags")
        if any("low-stimulation" in reason for reason in candidate.reasons):
            notes.append("was checked against low-stimulation safety")
        if any("vocalness feature supports" in reason for reason in candidate.reasons):
            notes.append("has low-vocal evidence")
        if any("uplift phase" in reason for reason in candidate.reasons):
            notes.append("fits the gentle-lift target")
        if not notes:
            notes.append("matches this phase better than nearby candidates")
        if len(notes) == 1:
            return f"Good fit for {candidate.phase.label}; {notes[0]}."
        return f"Good fit for {candidate.phase.label}, but {notes[0]}; {notes[1]}."

    def help_panel_text(self) -> str:
        """Return the full TUI keyboard help text."""

        return "\n".join(
            [
                "Help",
                "Playback",
                "Space / p  play / pause / resume",
                "Left/Right seek back / forward 10 seconds",
                "Up/Down    volume up / down 5%",
                ">          next track, no feedback",
                "<          previous track, no feedback",
                "x          stop playback",
                "m          playback mode",
                "Feedback",
                "s          skip and record negative feedback",
                "l          like current track",
                "v          prefer less vocals",
                "+          too loud; lower upcoming energy",
                "-          too slow; raise upcoming energy",
                "Tools",
                "c          Setup / Getting Started",
                "Ctrl+L     listening history",
                "d          Data & Privacy (outside Request or Memory input)",
                "Ctrl+O     memory notes panel (Control + letter o, not zero)",
                "Ctrl+S     save memory locally",
                "Ctrl+Enter save memory and update profile",
                "Ctrl+P     show memory profile",
                "Ctrl+G     generate memory suggestions",
                "j / k      move suggestion selection",
                "Enter      apply selected suggestion",
                "i          AI Assist status",
                "e          expand/collapse events",
                "w          write full why to events",
                "a / r      Codex audit / rerank preview",
                "/ / n      prompt / new request",
                "Esc        finish prompt editing without submitting",
                "Theme",
                "t          cycle Warmline / Midnight / High Contrast / Solarized / Catppuccin / Dracula / Jukebox",
                "q          quit when prompt is not focused",
                "Ctrl+Q     quit anytime",
            ]
        )

    def ai_assist_panel_text(self) -> str:
        """Return a redacted AI Assist status explanation."""

        settings = config.load_config()
        try:
            provider = provider_config()
            provider_text = provider.provider
            provider_ready = provider.configured
            key_text = "configured" if provider_ready else f"missing {provider.api_key_env}"
        except ValueError:
            provider_ready = False
            provider_text = "invalid provider"
            key_text = "provider config invalid"
        will_call = "yes, on new prompts" if settings.experience.mode == "smart" and settings.privacy.send_to_llm and provider_ready else "no"
        return "\n".join(
            [
                "AI Assist",
                f"Status: {self.llm_status_label(settings)}",
                f"Provider: {provider_text}",
                f"Key: {key_text}",
                f"Will call LLM: {will_call}",
                "What it does: parse your listening intent.",
                "What it will not do: invent BPM, vocalness, tags, or track facts.",
                "Enable:",
                "uv run tonepath setup --preset smart --send-to-llm",
                "Check:",
                "uv run tonepath llm doctor",
            ]
        )

    def why_panel_renderable(self) -> Text:
        """Return styled explanation preview content."""

        text = Text()
        for line in self.why_panel_text().splitlines():
            if line in {"Why", "Evidence", "Help", "Playback", "Feedback", "Tools", "Theme", "AI Assist"} or line.startswith("Missing evidence"):
                if text:
                    text.append("\n")
                if line == "Evidence":
                    style = self.palette.secondary
                elif line.startswith("Missing evidence"):
                    style = self.palette.muted
                else:
                    style = self.palette.primary
                text.append(line, style=f"bold {style}")
                continue
            text.append("\n")
            if "caution" in line or "risk" in line:
                style = self.palette.warning
            elif line.startswith("Confidence") or line.startswith("BPM") or line.startswith("Vocalness"):
                style = self.palette.muted
            else:
                style = self.palette.text
            text.append(line, style=style)
        return text

    def render_intake(self) -> None:
        """Render the no-session intake state."""

        self.refresh_readiness()
        if readiness_blocks_session(self.readiness):
            guidance = f"Library is not ready: {self.readiness}. {self.readiness_action}"
        elif not self.model_runtime_ready:
            guidance = "Ready for playback. Better vocalness is available after `uv run tonepath models setup essentia-tf`."
        else:
            guidance = "Ready for playback. How are you feeling? What should music help you become?"
        self.apply_panel_titles()
        if self.setup_prepare_busy:
            status_extra = "Preparing library"
        else:
            status_extra = "Planning next path" if self.request_busy else "Enter to plan"
        self.query_one("#status-bar", Static).update(self.status_bar_text(extra=status_extra))
        self.query_one("#timeline", Static).update("No session yet · feeling → path → feedback → memory")
        self.query_one("#command-bar", Static).update(self.command_bar_renderable(prompt_focused=True))
        self.query_one("#now-playing", Static).update(
            Text.assemble(
                ("● No session yet\n", f"bold {self.palette.primary}"),
                (f"{guidance}\n", self.palette.text),
                ("Example:\n", self.palette.muted),
                (PROMPT_PLACEHOLDER, self.palette.primary),
            )
        )
        self.refresh_right_panel()
        table = self.query_one("#queue", DataTable)
        table.clear(columns=False)

    def energy_text(self, track_id: int | None) -> str:
        """Return a compact energy value for queue rows."""

        if self.store is None or track_id is None:
            return "--"
        features = self.store.get_features(track_id)
        if features is None or features.energy is None:
            return "--"
        return f"{features.energy:.2f}"

    def fit_label(self, candidate: CandidateScore) -> str:
        """Return a short queue label that explains candidate fit at a glance."""

        features = self.store.get_features(candidate.track.id) if self.store is not None and candidate.track.id else None
        labels: list[str] = []
        if any("semantic risk" in reason for reason in candidate.reasons):
            labels.append("caution")
        if any("vocalness feature supports" in reason for reason in candidate.reasons):
            labels.append("low-vocal")
        if any("low-stimulation" in reason or "sleep/calm" in reason for reason in candidate.reasons):
            labels.append("calm-safe")
        if any("uplift phase" in reason for reason in candidate.reasons):
            labels.append("uplift")
        if features is None:
            labels.append("low-info")
        elif (features.energy is not None and features.energy >= 0.65) or (features.bpm is not None and features.bpm >= 140.0):
            labels.append("high-energy")
        if not labels:
            labels.append("fit")
        return " ".join(labels[:2])

    def library_count(self) -> int:
        """Return the number of scanned tracks available to the TUI."""

        if self.store is None:
            return 0
        return len(self.store.list_tracks())

    def missing_feature_count(self) -> int:
        """Return the number of tracks without analyzed features."""

        self.refresh_readiness()
        if self.library_status is None:
            return 0
        return self.library_status.missing_features

    def refresh_readiness(self) -> None:
        """Refresh cached library readiness state for TUI decisions."""

        if self.store is None:
            self.library_status = None
            self.readiness = "Needs setup"
            self.readiness_action = "Run `uv run tonepath setup --preset private`."
            return
        settings = config.load_config()
        self.library_status = library_status(self.store)
        self.readiness = readiness_label(self.library_status, self.model_runtime_ready, settings)
        self.readiness_action = status_next_action(self.library_status, self.model_runtime_ready, settings)

    def privacy_text(self) -> str:
        """Return local mode, readiness, and external-capability status lines."""

        settings = config.load_config()
        llm_state = self.llm_status_label(settings)
        model_state = "Model Ready" if self.model_runtime_ready else "Model Missing"
        codex_state = "Codex Available" if shutil.which("codex") else "Codex Optional"
        return "\n".join(
            [
                f"✓ {self.experience_label()}",
                f"✓ {self.readiness}",
                f"✓ {model_state}",
                f"✓ {llm_state}",
                f"✓ {codex_state}",
            ]
        )

    def llm_status_label(self, settings: config.TonepathConfig) -> str:
        """Return a redacted LLM status for the mode/privacy badge."""

        if not settings.privacy.send_to_llm:
            return "AI Assist Off"
        try:
            provider = provider_config()
        except ValueError:
            return "AI Assist Provider Invalid"
        return f"AI Assist Ready: {provider.provider}" if provider.configured else f"AI Assist Missing Key: {provider.provider}"

    def status_bar_text(self, extra: str | None = None) -> str:
        """Return a compact session status line."""

        settings = config.load_config()
        llm_state = self.status_ai_label(settings)
        model_state = "Model ✓" if self.model_runtime_ready else "Model missing"
        parts = [
            f"{playback_symbol(self.playback_status)} {self.playback_status}",
            self.playback_mode,
            self.experience_label(),
            model_state,
            llm_state,
            f"{self.library_count()} tracks",
        ]
        if extra:
            parts.append(extra)
        return " · ".join(parts)

    def command_bar_renderable(self, prompt_focused: bool | None = None) -> Text:
        """Return the persistent player command bar."""

        if prompt_focused is None:
            prompt_focused = bool(getattr(self.query_one("#prompt-input", Input), "has_focus", False))
        commands = [
            ("Space", "Play/Pause"),
            (">", "Next"),
            ("<", "Prev"),
            ("s", "Skip"),
            ("l", "Like"),
            ("m", "Mode"),
            ("Ctrl+L", "History"),
            ("d", "Data"),
            ("Ctrl+O", "Memory"),
            ("t", "Theme"),
            ("?", "Help"),
        ]
        if not config.config_path().exists() or readiness_blocks_session(self.readiness):
            commands.insert(6, ("c", "Setup"))
        if self.right_panel == "memory":
            commands.insert(0, ("Ctrl+S", "Save"))
            commands.insert(1, ("Ctrl+Enter", "Save+Learn"))
        elif self.right_panel == "memory_suggestions":
            commands.insert(0, ("Enter", "Apply"))
            commands.insert(1, ("j/k", "Move"))
        if prompt_focused:
            commands.insert(0, ("Enter", "Submit"))
            commands.insert(1, ("Esc", "Done"))
            commands.insert(2, ("Ctrl+Q", "Quit"))
        text = Text()
        for index, (key, label) in enumerate(commands):
            if index:
                text.append("  ", style=self.palette.muted)
            text.append(f" {key} ", style=f"bold {self.palette.background} on {self.palette.primary}")
            text.append(f" {label}", style=f"bold {self.palette.text}")
        return text

    def status_ai_label(self, settings: config.TonepathConfig) -> str:
        """Return a compact AI Assist label for the status bar."""

        if not settings.privacy.send_to_llm:
            return "AI off"
        try:
            provider = provider_config()
        except ValueError:
            return "AI invalid"
        return f"AI {provider.provider}" if provider.configured else f"AI key missing: {provider.provider}"

    def experience_label(self) -> str:
        """Return the active normal-user experience label."""

        return config.load_config().experience.mode.title()

    def install_themes(self) -> None:
        """Register Tonepath palettes with Textual's theme system."""

        for palette in PALETTES:
            self.register_theme(
                Theme(
                    name=palette.key,
                    primary=palette.primary,
                    secondary=palette.secondary,
                    warning=palette.warning,
                    success=palette.success,
                    accent=palette.accent,
                    foreground=palette.text,
                    background=palette.background,
                    surface=palette.surface,
                    panel=palette.panel,
                    dark=palette.dark,
                )
            )

    def latest_codex_preview(self) -> list[str] | None:
        """Return a compact rerank preview from the newest Codex audit for this session."""

        if self.runner is None:
            return None
        current_prompt = self.runner.active_plan().request.prompt
        try:
            payload = evaluate_rerank(current_prompt)
        except RuntimeError:
            return None
        if not payload.get("found"):
            return None
        counts = payload.get("counts")
        details = payload.get("details")
        if not isinstance(counts, dict) or not isinstance(details, list):
            return ["Latest Codex audit result is unreadable."]
        lines = [
            (
                f"Rerank preview: keep {counts.get('keep', 0)} · demote {counts.get('demote', 0)} · "
                f"reject {counts.get('reject', 0)} · not audited {counts.get('not_audited', 0)}"
            )
        ]
        for row in details[:3]:
            if not isinstance(row, dict):
                continue
            track = row.get("track")
            if not isinstance(track, dict):
                continue
            title = track.get("title") or "unknown"
            artist = track.get("artist") or "unknown"
            lines.append(f"{row.get('decision', 'not_audited')}: {title} - {artist} · {row.get('suggested_action')}")
        return lines

    def latest_codex_summary(self) -> str | None:
        """Return the first line of the newest matching Codex rerank preview."""

        preview = self.latest_codex_preview()
        return None if preview is None else preview[0]

    def show_empty_library(self) -> None:
        """Render setup guidance when no local tracks are available."""

        self.query_one("#timeline", Static).update("Tonepath: setup required")
        self.playback_status = "No tracks"
        self.query_one("#status-bar", Static).update(self.status_bar_text(extra="setup required"))
        self.query_one("#prompt-input", Input).value = ""
        self.query_one("#now-playing", Static).update(
            "No scanned tracks.\n\nPress c for Setup, or run:\nuv run tonepath setup"
        )
        self.refresh_right_panel()
        self.query_one("#command-bar", Static).update(self.command_bar_renderable(prompt_focused=True))
        self.log_event("No local tracks found.")

    def log_event(self, message: str) -> None:
        """Append an event to the bottom log panel."""

        self.query_one("#event-log", RichLog).write(message)

    def apply_panel_titles(self) -> None:
        """Apply stable panel titles to the TUI widgets."""

        self.query_one("#now-playing", Static).border_title = "Now"
        self.query_one("#queue", DataTable).border_title = "Queue"
        titles = {"why": "Why", "help": "Help", "ai_assist": "AI Assist"}
        self.query_one("#why-panel", Static).border_title = titles.get(self.right_panel, "Why")
        memory_input = self.query_one("#memory-input", TextArea)
        memory_input.border_title = "Memory"
        memory_input.border_subtitle = self.memory_status_message if self.right_panel == "memory" else ""
        self.query_one("#memory-profile", Static).border_title = "Memory Profile"
        self.query_one("#memory-suggestions", DataTable).border_title = "Suggestions"
        self.query_one("#event-log", RichLog).border_title = "Events expanded" if self.events_expanded else "Events"
        self.query_one("#prompt-input", Input).border_title = "Request"
        self.query_one("#timeline", Static).border_title = "Path"

    def show_right_panel(self) -> None:
        """Show only the widget backing the active right-panel mode."""

        memory_input = self.query_one("#memory-input", TextArea)
        if self.right_panel != "memory" and bool(getattr(memory_input, "display", False)):
            self.memory_draft = memory_input.text
        self.query_one("#why-panel", Static).display = self.right_panel in {"why", "help", "ai_assist"}
        memory_input.display = self.right_panel == "memory"
        self.query_one("#memory-profile", Static).display = self.right_panel == "memory_profile"
        self.query_one("#memory-suggestions", DataTable).display = self.right_panel == "memory_suggestions"

    def refresh_right_panel(self) -> None:
        """Refresh the active right-panel widget."""

        self.apply_panel_titles()
        self.show_right_panel()
        if self.right_panel in {"why", "help", "ai_assist"}:
            self.query_one("#why-panel", Static).update(self.why_panel_renderable())
        elif self.right_panel == "memory":
            memory_input = self.query_one("#memory-input", TextArea)
            if bool(getattr(memory_input, "has_focus", False)):
                self.memory_draft = memory_input.text
            elif memory_input.text != self.memory_draft:
                memory_input.load_text(self.memory_draft)
        elif self.right_panel == "memory_profile":
            self.query_one("#memory-profile", Static).update(self.memory_profile_renderable())
        elif self.right_panel == "memory_suggestions":
            self.refresh_memory_suggestions()

    def sync_memory_draft(self) -> None:
        """Copy the current memory text area content into app state."""

        try:
            self.memory_draft = self.query_one("#memory-input", TextArea).text
        except Exception:
            return

    def save_memory_draft(self, allow_empty: bool = False) -> bool:
        """Persist the current memory draft to local memory logs."""

        self.sync_memory_draft()
        body = self.memory_draft.strip()
        if not body:
            if not allow_empty:
                self.memory_status_message = "Tree-hole draft is empty."
                self.right_panel = "memory"
                self.log_event(self.memory_status_message)
                self.refresh_session_view()
            return False
        try:
            record = add_memory_log(body, source="tui")
        except (OSError, ValueError) as exc:
            self.memory_status_message = str(exc)
            self.right_panel = "memory"
            self.log_event(self.memory_status_message)
            self.refresh_session_view()
            return False
        self.memory_draft = ""
        self.query_one("#memory-input", TextArea).load_text("")
        self.log_event(f"Memory saved: {record['id']}. Current queue is unchanged.")
        return True

    def llm_ready(self, settings: config.TonepathConfig) -> bool:
        """Return whether the TUI may call its configured LLM."""

        if not settings.privacy.send_to_llm:
            return False
        try:
            provider = provider_config()
        except ValueError:
            return False
        return provider.configured

    def memory_profile_renderable(self) -> Text:
        """Return styled memory profile text plus current TUI status."""

        text = Text()
        if self.memory_status_message:
            text.append("Status\n", style=f"bold {self.palette.primary}")
            text.append(f"{self.memory_status_message}\n\n", style=self.palette.text)
        text.append(memory_profile_text(), style=self.palette.text)
        return text

    def load_memory_suggestions(self) -> None:
        """Load pending profile suggestions for the TUI memory panel."""

        self.memory_suggestions = self.memory_suggestion_items()
        if self.selected_memory_suggestion_index >= len(self.memory_suggestions):
            self.selected_memory_suggestion_index = max(len(self.memory_suggestions) - 1, 0)

    def memory_suggestion_items(self) -> list[dict[str, object]]:
        """Return grouped and individual pending suggestions for display."""

        items: list[dict[str, object]] = []
        for group in pending_suggestion_groups():
            items.append(
                {
                    "kind": "group",
                    "id": str(group.get("group_id", "--")),
                    "scope": str(group.get("scope", "--")),
                    "confidence": str(group.get("confidence", "--")),
                    "rationale": str(group.get("rationale") or group.get("hint") or ""),
                }
            )
        grouped_ids: set[str] = set()
        for group in pending_suggestion_groups():
            ids = group.get("suggestion_ids")
            if isinstance(ids, list):
                grouped_ids.update(str(item) for item in ids)
        for suggestion in list_pending_suggestions():
            suggestion_id = str(suggestion.get("suggestion_id", "--"))
            if suggestion_id in grouped_ids:
                continue
            items.append(
                {
                    "kind": "rule",
                    "id": suggestion_id,
                    "scope": str(suggestion.get("scope", "--")),
                    "confidence": str(suggestion.get("confidence", "--")),
                    "rationale": str(suggestion.get("rationale", "")),
                }
            )
        return items

    def refresh_memory_suggestions(self) -> None:
        """Render memory-derived suggestions in the right-panel table."""

        table = self.query_one("#memory-suggestions", DataTable)
        table.clear(columns=False)
        items = self.memory_suggestion_items()
        self.memory_suggestions = items
        if not items:
            table.add_row(
                queue_cell("-", palette=self.palette),
                queue_cell("none", palette=self.palette),
                queue_cell("--", palette=self.palette),
                queue_cell("--", palette=self.palette),
                queue_cell(self.memory_status_message or "No pending memory suggestions.", palette=self.palette),
            )
            return
        self.selected_memory_suggestion_index = min(self.selected_memory_suggestion_index, len(items) - 1)
        for index, item in enumerate(items):
            current = index == self.selected_memory_suggestion_index
            table.add_row(
                queue_cell("▶" if current else str(index + 1), current=current, palette=self.palette),
                queue_cell(str(item["kind"]), current=current, palette=self.palette),
                queue_cell(str(item["scope"]), current=current, palette=self.palette),
                queue_cell(str(item["confidence"]), current=current, palette=self.palette),
                queue_cell(truncate(str(item["rationale"]), 54), current=current, palette=self.palette),
            )


def run_tui(prompt: str | None = None) -> None:
    """Run the Tonepath terminal interface."""

    TonepathApp(prompt=prompt).run()


def truncate(value: str, limit: int) -> str:
    """Truncate long labels for stable terminal layout."""

    if len(value) <= limit:
        return value
    return f"{value[: max(limit - 1, 0)]}…"


def queue_marker(position: str) -> str:
    """Return a compact queue marker for the current and upcoming tracks."""

    if position == "now":
        return "▶"
    return position.replace("+", "")


def queue_cell(value: str, current: bool = False, align: str | None = None, palette: TonepathPalette | None = None) -> Text:
    """Return a styled queue table cell."""

    active_palette = palette or PALETTE_BY_KEY["warmline"]
    style = f"bold {active_palette.primary}" if current else active_palette.muted
    return Text(value, style=style, justify=align)


def fit_cell(value: str, current: bool = False, palette: TonepathPalette | None = None) -> Text:
    """Return a queue fit label with compact semantic color coding."""

    active_palette = palette or PALETTE_BY_KEY["warmline"]
    text = Text(justify=None)
    labels = value.split()
    for index, label in enumerate(labels):
        if index:
            text.append(" ")
        style = fit_label_style(label, active_palette, current=current)
        text.append(label, style=style)
    return text


def fit_label_style(label: str, palette: TonepathPalette, current: bool = False) -> str:
    """Return the rich style for a compact queue fit label."""

    prefix = "bold " if current else ""
    if label == "caution" or label == "high-energy":
        return f"{prefix}{palette.warning}"
    if label in {"low-vocal", "calm-safe"}:
        return f"{prefix}{palette.secondary}"
    if label == "uplift":
        return f"{prefix}{palette.success}"
    if label == "low-info":
        return f"{prefix}{palette.muted}"
    return f"{prefix}{palette.primary if current else palette.text}"


def playback_symbol(status: str) -> str:
    """Return a stable text-friendly playback symbol."""

    if status == "Playing":
        return "●"
    if status == "Paused":
        return "Ⅱ"
    if status in {"Stopped", "Finished"}:
        return "■"
    if status.startswith("Need") or status == "No tracks":
        return "!"
    return "○"


def format_clock(seconds: float) -> str:
    """Return a compact m:ss clock label."""

    total = max(int(seconds), 0)
    minutes, remainder = divmod(total, 60)
    return f"{minutes}:{remainder:02d}"


def progress_bar(elapsed: float, duration: float, width: int = 12) -> str:
    """Return a fixed-width progress bar for reported playback progress."""

    if duration <= 0:
        return "─" * width
    ratio = min(max(elapsed / duration, 0.0), 1.0)
    filled = min(max(round(ratio * width), 0), width)
    return "━" * filled + "─" * (width - filled)


def pulse_meter(level: float, tick: int, width: int = 8) -> str:
    """Return a subtle animated energy pulse indicator."""

    glyphs = ("▁", "▂", "▃", "▄", "▅", "▆")
    bounded = min(max(level, 0.0), 1.0)
    peak = min(max(round(bounded * (len(glyphs) - 1)), 1), len(glyphs) - 1)
    values = []
    for index in range(width):
        wave = (index + tick) % 4
        offset = 1 if wave in {1, 2} else 0
        values.append(glyphs[min(peak + offset, len(glyphs) - 1)])
    return "".join(values)


def bpm_text(bpm: float | None) -> str:
    """Return a compact BPM label without inventing missing tempo."""

    if bpm is None:
        return "unknown"
    return f"{bpm:.0f}"


def vocalness_text(vocalness: float | None) -> str:
    """Return a compact vocalness label without inventing missing vocals."""

    if vocalness is None:
        return "unknown"
    return f"{vocalness:.2f}"


def energy_meter(energy: float | None) -> str:
    """Return a five-step static energy strip."""

    if energy is None:
        return "▯▯▯▯▯"
    filled = min(max(round(energy * 5), 0), 5)
    return "▮" * filled + "▯" * (5 - filled)


def confidence_label(confidence: str) -> str:
    """Return a compact confidence label for narrow queue cells."""

    if confidence == "medium":
        return "med"
    return confidence


def should_auto_open_setup() -> bool:
    """Return whether this looks like a truly new home without saved setup."""

    if config.config_path().exists():
        return False
    home = config.app_home()
    if not home.exists():
        return True
    try:
        return not any(home.iterdir())
    except OSError:
        return False
