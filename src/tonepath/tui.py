"""Textual terminal interface for Tonepath."""

from __future__ import annotations

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import FeedbackType
from tonepath.session import SessionRunner

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import DataTable, Footer, Header, RichLog, Static
except ImportError as exc:  # pragma: no cover - exercised before dependency install
    raise RuntimeError("Textual is not installed. Run `uv sync` before launching the TUI.") from exc


DEFAULT_TUI_PROMPT = "from irritated to focused in 30 minutes, no vocals"


class TonepathApp(App[None]):
    """Terminal session screen for a local Tonepath listening path."""

    CSS = """
    Screen {
        layout: vertical;
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
        width: 58%;
        min-width: 50;
    }

    #right-pane {
        width: 42%;
        min-width: 38;
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
        ("s", "skip", "Skip"),
        ("l", "like", "Like"),
        ("v", "no_vocals", "No vocals"),
        ("+", "too_loud", "Too loud"),
        ("-", "too_slow", "Too slow"),
        ("w", "why", "Why"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, prompt: str = DEFAULT_TUI_PROMPT) -> None:
        super().__init__()
        self.prompt = prompt
        self.store: TonepathStore | None = None
        self.runner: SessionRunner | None = None

    def compose(self) -> ComposeResult:
        """Compose the terminal product surface with Textual built-ins."""

        yield Header()
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
        table.add_columns("Pos", "Phase", "Track", "Conf", "Score")

        if not self.store.list_tracks():
            self.show_empty_library()
            return

        self.runner = SessionRunner(self.store, self.prompt)
        self.log_event(f"Started local session: {self.prompt}")
        self.refresh_session_view()

    def on_unmount(self) -> None:
        """Close local storage when the terminal app exits."""

        if self.store is not None:
            self.store.close()

    def action_skip(self) -> None:
        """Skip the current candidate and refresh upcoming recommendations."""

        self.apply_feedback("skip")

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
            self.log_event("No active session.")
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

    def refresh_session_view(self) -> None:
        """Refresh timeline, queue, why panel, and privacy badge."""

        if self.runner is None:
            return

        self.query_one("#timeline", Static).update(self.timeline_text())
        self.query_one("#now-playing", Static).update(self.now_playing_text())
        self.query_one("#why-panel", Static).update(self.runner.current_explanation())
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
                candidate.track.title or "unknown",
                candidate.confidence,
                f"{candidate.score:.2f}",
            )

    def timeline_text(self) -> str:
        """Return a compact path timeline for the current plan."""

        if self.runner is None:
            return "Tonepath"
        phase_labels = " -> ".join(phase.label for phase in self.runner.active_plan().phases)
        request = self.runner.active_plan().request
        return f"{request.source_state} -> {phase_labels} -> {request.target_state}"

    def now_playing_text(self) -> str:
        """Return the now-playing panel text."""

        if self.runner is None:
            return "No active session."
        candidate = self.runner.current()
        if candidate is None:
            return "Queue is empty. Run `tonepath scan` to add local music."
        return "\n".join(
            [
                "Now Playing",
                f"Track: {candidate.track.title or 'unknown'}",
                f"Artist: {candidate.track.artist or 'unknown'}",
                f"Phase: {candidate.phase.label}",
                f"Confidence: {candidate.confidence}",
            ]
        )

    def privacy_text(self) -> str:
        """Return the local privacy badge text."""

        return "\n".join(
            [
                "Privacy",
                "Network: offline by default",
                f"DB: {config.db_path()}",
                "No web/LLM enrichment during playback.",
            ]
        )

    def show_empty_library(self) -> None:
        """Render setup guidance when no local tracks are available."""

        self.query_one("#timeline", Static).update("Tonepath: local library required")
        self.query_one("#now-playing", Static).update(
            "No scanned tracks.\n\nRun:\nuv run tonepath config add-music-dir /path/to/music\nuv run tonepath scan"
        )
        self.query_one("#why-panel", Static).update("Why panel appears after a local session starts.")
        self.query_one("#privacy-badge", Static).update(self.privacy_text())
        self.log_event("No local tracks found.")

    def log_event(self, message: str) -> None:
        """Append an event to the bottom log panel."""

        self.query_one("#event-log", RichLog).write(message)


def run_tui(prompt: str = DEFAULT_TUI_PROMPT) -> None:
    """Run the Tonepath terminal interface."""

    TonepathApp(prompt=prompt).run()
