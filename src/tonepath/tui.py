"""Textual terminal interface for Tonepath."""

from __future__ import annotations

from typing import Any

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import FeedbackType
from tonepath.playback_controller import PlaybackController
from tonepath.session import SessionRunner

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise RuntimeError("Textual is not installed. Run `uv sync` before launching the TUI.") from exc


PROMPT_PLACEHOLDER = "我现在很烦，想半小时后进入写代码状态，不要人声"


class TonepathApp(App[None]):
    """Terminal session screen for a local Tonepath listening path."""

    TITLE = "Tonepath"
    SUB_TITLE = "Local state-transition player"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        height: 3;
        padding: 1 2;
        text-style: bold;
        background: $surface;
    }

    #prompt-input {
        height: 3;
        margin: 0 1;
    }

    #timeline {
        height: 3;
        padding: 1 2;
        text-style: bold;
        background: $surface;
    }

    #body {
        height: 1fr;
    }

    #left-pane {
        width: 64%;
        min-width: 50;
    }

    #right-pane {
        width: 36%;
        min-width: 28;
        border-left: solid $primary;
    }

    #now-playing,
    #privacy-badge,
    #why-panel {
        padding: 1 2;
    }

    #now-playing {
        height: 8;
        border-bottom: solid $primary;
    }

    #queue {
        height: 1fr;
    }

    #why-panel {
        height: 1fr;
        border-bottom: solid $primary;
    }

    #privacy-badge {
        height: 6;
    }

    #event-log {
        height: 7;
        border-top: solid $primary;
    }
    """

    BINDINGS = [
        Binding("/", "focus_prompt", "Prompt"),
        Binding("n", "new_prompt", "New"),
        Binding("space", "play", "Play"),
        Binding("p", "play", "Play", show=False),
        Binding("x", "stop_playback", "Stop"),
        Binding("s", "skip", "Skip"),
        Binding("l", "like", "Like"),
        Binding("v", "no_vocals", "No vocals"),
        Binding("+", "too_loud", "Quieter", show=False),
        Binding("-", "too_slow", "More energy", show=False),
        Binding("w", "why", "Why"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, prompt: str | None = None) -> None:
        super().__init__()
        self.initial_prompt = prompt
        self.store: TonepathStore | None = None
        self.runner: SessionRunner | None = None
        self.playback: PlaybackController | None = None
        self.playback_status = "Ready"
        self.playback_timer: Any | None = None

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
                yield Static("", id="privacy-badge")
        yield RichLog(id="event-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Load local state and render the first session view."""

        self.store = TonepathStore()
        table = self.query_one("#queue", DataTable)
        table.add_columns("#", "Phase", "Track", "Conf")

        if not self.store.list_tracks():
            self.show_empty_library()
            return

        self.playback = PlaybackController(self.store)
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

    def action_play(self) -> None:
        """Start playback for the current candidate."""

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            self.query_one("#prompt-input", Input).focus()
            return
        self.start_current_playback()

    def action_stop_playback(self) -> None:
        """Stop active playback without exiting the TUI."""

        if self.playback is None:
            self.log_event("No playback controller.")
            return
        stopped = self.playback.stop_current()
        self.playback_status = "Stopped"
        self.log_event("Stopped playback." if stopped else "No active Tonepath playback.")
        self.refresh_session_view()

    def action_focus_prompt(self) -> None:
        """Focus the prompt input bar."""

        self.query_one("#prompt-input", Input).focus()

    def action_new_prompt(self) -> None:
        """Return to the prompt intake state."""

        if self.playback is not None:
            self.playback.stop_current()
        self.runner = None
        self.playback_status = "Ready"
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

        if self.runner is None:
            self.log_event("Enter a listening goal first.")
            return
        self.log_event(self.runner.current_explanation())

    def apply_feedback(self, feedback_type: FeedbackType) -> None:
        """Apply one feedback action to the active session."""

        if self.runner is None:
            self.log_event("No active session.")
            return
        message = self.runner.apply_feedback(feedback_type)
        self.log_event(message)
        self.refresh_session_view()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Create a new local session when the prompt input is submitted."""

        if event.input.id != "prompt-input":
            return
        self.create_session(event.value)

    def create_session(self, prompt: str) -> None:
        """Create a local session from a user prompt and refresh the TUI."""

        cleaned = prompt.strip()
        if not cleaned:
            self.log_event("Prompt is empty. Type a listening goal first.")
            self.query_one("#prompt-input", Input).focus()
            return
        if self.store is None:
            self.log_event("Local store is unavailable.")
            return
        if self.playback is not None:
            self.playback.stop_current()
        self.runner = SessionRunner(self.store, cleaned)
        self.playback = self.playback or PlaybackController(self.store)
        self.playback_status = "Ready"
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = cleaned
        prompt_input.blur()
        self.log_event(f"Ready. Press Space to play. Session: {cleaned}")
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
        self.playback_status = "Playing"
        self.log_event(f"Playing: {track_label(candidate.track.title, candidate.track.path.name)}")
        self.ensure_playback_polling()
        self.refresh_session_view()

    def ensure_playback_polling(self) -> None:
        """Start a lightweight TUI-local poller for natural mpv exits."""

        if self.playback_timer is None:
            self.playback_timer = self.set_interval(0.5, self.poll_playback_finished)

    def poll_playback_finished(self) -> None:
        """Update local session state when mpv exits without an explicit stop."""

        if self.playback is None or not self.playback.finish_if_exited():
            return
        self.playback_status = "Finished"
        self.log_event("Playback finished.")
        self.refresh_session_view()

    def refresh_session_view(self) -> None:
        """Refresh timeline, queue, why panel, and privacy badge."""

        if self.runner is None:
            self.render_intake()
            return

        self.query_one("#status-bar", Static).update(f"Tonepath · {self.playback_status} · / prompt · n new")
        self.query_one("#timeline", Static).update(self.timeline_text())
        self.query_one("#now-playing", Static).update(self.now_playing_text())
        self.query_one("#why-panel", Static).update(self.why_panel_text())
        self.query_one("#privacy-badge", Static).update(self.privacy_text())
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
            table.add_row(
                position,
                candidate.phase.label,
                truncate(track_label(candidate.track.title, candidate.track.path.name), 30),
                candidate.confidence,
            )

    def timeline_text(self) -> str:
        """Return a compact path timeline for the current plan."""

        if self.runner is None:
            return "Tonepath"
        request = self.runner.active_plan().request
        labels = [phase.label for phase in self.runner.active_plan().phases]
        if labels and labels[-1] == request.target_state:
            labels = labels[:-1]
        path = " -> ".join([request.source_state, *labels, request.target_state])
        return f"{path} · {request.duration_sec // 60}m"

    def now_playing_text(self) -> str:
        """Return the now-playing panel text."""

        if self.runner is None:
            return "No active session."
        candidate = self.runner.current()
        if candidate is None:
            return "Queue is empty. Run `tonepath scan` to add local music."
        return "\n".join(
            [
                f"Status: {self.playback_status}",
                f"Track: {truncate(track_label(candidate.track.title, candidate.track.path.name), 42)}",
                f"Artist: {candidate.track.artist or 'unknown'}",
                f"Phase: {candidate.phase.label}",
                f"Confidence: {candidate.confidence}",
            ]
        )

    def why_panel_text(self) -> str:
        """Return a compact explanation preview for the right panel."""

        if self.runner is None:
            return "Why panel\n\nA verifiable explanation appears after Tonepath creates a listening path."
        candidate = self.runner.current()
        if candidate is None:
            return "Why panel\n\nNo current track."
        features = self.store.get_features(candidate.track.id) if self.store is not None and candidate.track.id else None
        energy = "unknown" if features is None or features.energy is None else f"{features.energy:.2f}"
        loudness = "unknown" if features is None or features.loudness is None else f"{features.loudness:.1f} dBFS"
        return "\n".join(
            [
                "Why this",
                f"Phase: {candidate.phase.label}",
                f"Target energy: {candidate.phase.target_energy:.2f}",
                f"Confidence: {candidate.confidence}",
                f"Energy: {energy}",
                f"Loudness: {loudness}",
                "BPM: unknown",
                "Vocalness: unknown",
                "",
                "Press w for full audit log.",
            ]
        )

    def render_intake(self) -> None:
        """Render the no-session intake state."""

        self.query_one("#status-bar", Static).update("Tonepath · Local state-transition player · offline")
        self.query_one("#timeline", Static).update("No session yet · type a listening goal and press Enter")
        self.query_one("#now-playing", Static).update(
            "\n".join(
                [
                    "No session yet",
                    "Use the prompt bar above.",
                    "Example:",
                    PROMPT_PLACEHOLDER,
                ]
            )
        )
        self.query_one("#why-panel", Static).update(
            self.why_panel_text()
        )
        self.query_one("#privacy-badge", Static).update(self.privacy_text())
        table = self.query_one("#queue", DataTable)
        table.clear(columns=False)

    def privacy_text(self) -> str:
        """Return the local privacy badge text."""

        return "\n".join(
            [
                "Privacy",
                "Offline by default",
                f"DB: {config.db_path().name}",
                "Audio files stay local.",
            ]
        )

    def show_empty_library(self) -> None:
        """Render setup guidance when no local tracks are available."""

        self.query_one("#timeline", Static).update("Tonepath: local library required")
        self.playback_status = "No tracks"
        self.query_one("#status-bar", Static).update("Tonepath · setup required")
        self.query_one("#prompt-input", Input).value = ""
        self.query_one("#now-playing", Static).update(
            "No scanned tracks.\n\nRun:\nuv run tonepath config add-music-dir /path/to/music\nuv run tonepath scan"
        )
        self.query_one("#why-panel", Static).update("Why panel appears after a local session starts.")
        self.query_one("#privacy-badge", Static).update(self.privacy_text())
        self.log_event("No local tracks found.")

    def log_event(self, message: str) -> None:
        """Append an event to the bottom log panel."""

        self.query_one("#event-log", RichLog).write(message)


def run_tui(prompt: str | None = None) -> None:
    """Run the Tonepath terminal interface."""

    TonepathApp(prompt=prompt).run()


def track_label(title: str | None, fallback: str) -> str:
    """Return a display-safe track label."""

    label = title or fallback
    return label.replace("(null)", "").strip() or fallback


def truncate(value: str, limit: int) -> str:
    """Truncate long labels for stable terminal layout."""

    if len(value) <= limit:
        return value
    return f"{value[: max(limit - 1, 0)]}…"
