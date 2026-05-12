"""Textual terminal interface for Tonepath."""

from __future__ import annotations

from typing import Any

from rich.text import Text

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
AMBER = "#d8a657"
AMBER_DIM = "#8f7242"
MUTED = "#a7afa5"
TEAL = "#6fb7a6"
TEXT = "#e6e0cf"


class TonepathApp(App[None]):
    """Terminal session screen for a local Tonepath listening path."""

    TITLE = "Tonepath"
    SUB_TITLE = "Local state-transition player"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen {
        layout: vertical;
        background: #101311;
        color: #e6e0cf;
    }

    #status-bar {
        height: 1;
        padding: 0 2;
        text-style: bold;
        background: #1d211d;
        color: #d8a657;
    }

    #prompt-input {
        height: 3;
        margin: 0 1;
        background: #171b18;
        border: round #8f7242;
        border-title-color: #d8a657;
        color: #e6e0cf;
    }

    #timeline {
        height: 3;
        padding: 1 2;
        text-style: bold;
        background: #151914;
        color: #d8a657;
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
    #privacy-badge,
    #why-panel {
        padding: 1 2;
        background: #171b18;
        border: round #3a4038;
        color: #e6e0cf;
    }

    #now-playing {
        height: 8;
        margin-bottom: 1;
        border-left: heavy #d8a657;
        border-title-color: #d8a657;
    }

    #queue {
        height: 1fr;
        background: #171b18;
        border: round #3a4038;
        border-title-color: #d8a657;
        color: #e6e0cf;
    }

    #why-panel {
        height: 1fr;
        margin-bottom: 1;
        border-title-color: #6fb7a6;
    }

    #privacy-badge {
        height: 7;
        border-title-color: #6fb7a6;
        color: #6fb7a6;
    }

    #event-log {
        height: 6;
        margin: 0 1;
        background: #151914;
        border: round #3a4038;
        border-title-color: #a7afa5;
        color: #a7afa5;
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
        self.apply_panel_titles()
        table = self.query_one("#queue", DataTable)
        table.add_columns("#", "Phase", "Track", "Energy", "Conf")

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

        self.query_one("#status-bar", Static).update(
            f"● {self.playback_status}   Local · {self.library_count()} tracks · offline · / prompt · n new"
        )
        self.query_one("#timeline", Static).update(self.timeline_text())
        self.query_one("#now-playing", Static).update(self.now_playing_renderable())
        self.query_one("#why-panel", Static).update(self.why_panel_renderable())
        self.query_one("#privacy-badge", Static).update(self.privacy_renderable())
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
                queue_cell(queue_marker(position), current=current, align="center"),
                queue_cell(candidate.phase.label, current=current),
                queue_cell(truncate(track_label(candidate.track.title, candidate.track.path.name), 28), current=current),
                queue_cell(self.energy_text(candidate.track.id), current=current),
                queue_cell(confidence_label(candidate.confidence), current=current),
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
        return "\n".join(
            [
                f"{self.playback_status} | {candidate.phase.label} | {confidence_label(candidate.confidence)}",
                truncate(track_label(candidate.track.title, candidate.track.path.name), 44),
                candidate.track.artist or "unknown artist",
                f"energy {energy} · loudness {loudness}",
            ]
        )

    def now_playing_renderable(self) -> Text:
        """Return styled now-playing content."""

        lines = self.now_playing_text().splitlines()
        text = Text()
        if not lines:
            return text
        text.append("● ", style=AMBER)
        text.append(lines[0], style=f"bold {AMBER}")
        for index, line in enumerate(lines[1:], start=1):
            text.append("\n")
            if index == 1:
                text.append(line, style=f"bold {TEXT}")
            elif index == 2:
                text.append(line, style=MUTED)
            else:
                text.append(line, style=AMBER_DIM)
        return text

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
                "Fit",
                f"{candidate.phase.label} · target {candidate.phase.target_energy:.2f}",
                "Evidence",
                f"conf {confidence_label(candidate.confidence)} · energy {energy}",
                f"loudness {loudness}",
                "Unknown",
                "BPM · vocalness",
            ]
        )

    def why_panel_renderable(self) -> Text:
        """Return styled explanation preview content."""

        text = Text()
        for line in self.why_panel_text().splitlines():
            if line in {"Fit", "Evidence", "Unknown"}:
                if text:
                    text.append("\n")
                style = TEAL if line == "Evidence" else AMBER if line == "Fit" else MUTED
                text.append(line, style=f"bold {style}")
                continue
            text.append("\n")
            style = MUTED if line == "BPM · vocalness" else TEXT
            text.append(line, style=style)
        return text

    def render_intake(self) -> None:
        """Render the no-session intake state."""

        self.query_one("#status-bar", Static).update(
            f"● Ready   Local · {self.library_count()} tracks · offline · Enter to plan"
        )
        self.query_one("#timeline", Static).update("No session yet · type a listening goal and press Enter")
        self.query_one("#now-playing", Static).update(
            Text.assemble(
                ("● No session yet\n", f"bold {AMBER}"),
                ("Use the prompt bar above.\n", TEXT),
                ("Example:\n", MUTED),
                (PROMPT_PLACEHOLDER, AMBER_DIM),
            )
        )
        self.query_one("#why-panel", Static).update(
            self.why_panel_renderable()
        )
        self.query_one("#privacy-badge", Static).update(self.privacy_renderable())
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

    def library_count(self) -> int:
        """Return the number of scanned tracks available to the TUI."""

        if self.store is None:
            return 0
        return len(self.store.list_tracks())

    def privacy_text(self) -> str:
        """Return the local privacy badge text."""

        return "\n".join(
            [
                "✓ offline",
                f"✓ {config.db_path().name}",
                "✓ audio local",
            ]
        )

    def privacy_renderable(self) -> Text:
        """Return styled local privacy badge content."""

        text = Text()
        for index, line in enumerate(self.privacy_text().splitlines()):
            if index:
                text.append("\n")
            text.append(line, style=f"bold {TEAL}" if index == 0 else TEAL)
        return text

    def show_empty_library(self) -> None:
        """Render setup guidance when no local tracks are available."""

        self.query_one("#timeline", Static).update("Tonepath: local library required")
        self.playback_status = "No tracks"
        self.query_one("#status-bar", Static).update("● No tracks   Local · 0 tracks · offline · setup required")
        self.query_one("#prompt-input", Input).value = ""
        self.query_one("#now-playing", Static).update(
            "No scanned tracks.\n\nRun:\nuv run tonepath config add-music-dir /path/to/music\nuv run tonepath scan"
        )
        self.query_one("#why-panel", Static).update("Why panel appears after a local session starts.")
        self.query_one("#privacy-badge", Static).update(self.privacy_renderable())
        self.log_event("No local tracks found.")

    def log_event(self, message: str) -> None:
        """Append an event to the bottom log panel."""

        self.query_one("#event-log", RichLog).write(message)

    def apply_panel_titles(self) -> None:
        """Apply stable panel titles to the TUI widgets."""

        self.query_one("#now-playing", Static).border_title = "Now"
        self.query_one("#queue", DataTable).border_title = "Queue"
        self.query_one("#why-panel", Static).border_title = "Why"
        self.query_one("#privacy-badge", Static).border_title = "Local Privacy"
        self.query_one("#event-log", RichLog).border_title = "Events"
        self.query_one("#prompt-input", Input).border_title = "Request"


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


def queue_marker(position: str) -> str:
    """Return a compact queue marker for the current and upcoming tracks."""

    if position == "now":
        return "▶"
    return position.replace("+", "")


def queue_cell(value: str, current: bool = False, align: str | None = None) -> Text:
    """Return a styled queue table cell."""

    style = f"bold {AMBER}" if current else MUTED
    return Text(value, style=style, justify=align)


def confidence_label(confidence: str) -> str:
    """Return a compact confidence label for narrow queue cells."""

    if confidence == "medium":
        return "med"
    return confidence
