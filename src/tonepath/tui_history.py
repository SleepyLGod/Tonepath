"""TUI-only saved-session browsing and exact replay helpers."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import DataTable, Header, Static

from tonepath.db import TonepathStore
from tonepath.history import (
    HistoryQueueItem,
    HistoryRecord,
    HistorySession,
    ReplayPreparation,
    create_replay_session,
    list_history,
    load_history,
    prepare_replay,
)
from tonepath.session import SessionRunner


@dataclass(frozen=True)
class HistoryLoadResult:
    """Exact replay state returned to the player after History closes."""

    source_session_id: int
    runner: ReplaySessionRunner
    omitted: tuple[HistoryQueueItem, ...]


@dataclass(frozen=True)
class HistoryRerunRequest:
    """Original Request selected for a fresh recommendation path."""

    source_session_id: int
    prompt: str


class ReplaySessionRunner(SessionRunner):
    """Run an exact historical queue without invoking the selector at load time."""

    def __init__(self, store: TonepathStore, replay: ReplayPreparation) -> None:
        self.store = store
        self.prompt = replay.plan.request.prompt
        self.limit_per_phase = 2
        self.base_plan = replay.plan
        self.session_id = create_replay_session(store, replay)
        self.current_index = 0
        self.energy_delta = 0.0
        self.force_no_vocals = replay.plan.request.no_vocals
        self.queue = list(replay.candidates)


class HistoryScreen(Screen[HistoryLoadResult | HistoryRerunRequest | None]):
    """Full-screen browser for played or bookmarked listening paths."""

    CSS = """
    HistoryScreen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    #history-heading {
        height: 3;
        padding: 1 2 0 2;
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #history-body {
        height: 1fr;
        padding: 1;
    }

    #history-list {
        width: 48%;
        min-width: 48;
        margin-right: 1;
        background: $panel;
        border: round $surface;
        border-title-color: $primary;
    }

    #history-list .datatable--header {
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #history-list .datatable--cursor {
        text-style: bold;
        color: $foreground;
        background: $primary;
    }

    #history-details {
        width: 52%;
        padding: 1 2;
        background: $panel;
        border: round $surface;
        border-title-color: $secondary;
    }

    #history-command-bar {
        height: 1;
        padding: 0 1;
        color: $foreground;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "close_history", "Close", show=False, priority=True),
        Binding("escape", "close_history", "Close", show=False, priority=True),
        Binding("j", "next_history", "Next", show=False),
        Binding("k", "previous_history", "Previous", show=False),
        Binding("space", "history_space", "Player only", show=False, priority=True),
        Binding("enter", "load_history", "Load", show=False, priority=True),
        Binding("r", "rerun_history", "Run Request", show=False, priority=True),
    ]

    def __init__(self, store: TonepathStore) -> None:
        super().__init__()
        self.store = store
        self.sessions: list[HistorySession] = []
        self.replay_statuses: dict[int, str] = {}
        self.selected_session_id: int | None = None
        self.status_message = ""

    def compose(self) -> ComposeResult:
        """Compose the history browser using the current Tonepath theme."""

        yield Header()
        yield Static("Listening History", id="history-heading")
        with Horizontal(id="history-body"):
            yield DataTable(id="history-list")
            yield Static("", id="history-details")
        yield Static(
            " ↑/↓ or j/k  Browse   Enter  Open   r  Run Request   Esc or Ctrl+L  Back ",
            id="history-command-bar",
        )

    def on_mount(self) -> None:
        """Load visible history without changing the active player state."""

        table = self.query_one("#history-list", DataTable)
        table.border_title = "Played / Saved Paths"
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Saved", "Replay", "When", "Request", "Transition", "Tracks")
        details = self.query_one("#history-details", Static)
        details.border_title = "Path Details"
        self.sessions = list_history(self.store)
        for session in self.sessions:
            replay_status = history_replay_status(self.store, session)
            self.replay_statuses[session.id] = replay_status
            saved_label = ""
            if session.saved:
                saved_label = f"★ {session.bookmark_name or 'Saved'}"
            table.add_row(
                _truncate(saved_label, 18),
                _replay_status_text(replay_status),
                _short_timestamp(session.started_at),
                _truncate(session.prompt, 30),
                _truncate(f"{session.source_state} -> {session.target_state}", 24),
                str(session.queue_count),
                key=str(session.id),
            )
        if self.sessions:
            self.selected_session_id = self.sessions[0].id
            table.move_cursor(row=0)
        self.refresh_details()
        table.focus()

    def action_close_history(self) -> None:
        """Return to the player without changing its queue or playback."""

        self.dismiss(None)

    def action_next_history(self) -> None:
        """Move to the next history record."""

        table = self.query_one("#history-list", DataTable)
        table.action_cursor_down()

    def action_previous_history(self) -> None:
        """Move to the previous history record."""

        table = self.query_one("#history-list", DataTable)
        table.action_cursor_up()

    def action_history_space(self) -> None:
        """Explain that Space remains a player control outside History."""

        status = self.replay_statuses.get(self.selected_session_id or -1)
        if status in {"Legacy", "Unavailable"}:
            self.status_message = (
                "Space does not act in History. Press Enter to run this Request again."
            )
        else:
            self.status_message = (
                "Space does not act in History. Press Enter to load, then Space in the player."
            )
        self.refresh_details()

    def action_load_history(self) -> None:
        """Prepare the selected exact queue, keeping failures inside History."""

        if self.selected_session_id is None:
            self.status_message = "No played or saved paths are available."
            self.refresh_details()
            return
        replay_status = self.replay_statuses.get(self.selected_session_id)
        if replay_status in {"Legacy", "Unavailable"}:
            self.action_rerun_history()
            return
        try:
            replay = prepare_replay(self.store, self.selected_session_id)
            runner = ReplaySessionRunner(self.store, replay)
        except RuntimeError as exc:
            self.status_message = str(exc)
            self.refresh_details()
            return
        self.dismiss(
            HistoryLoadResult(
                source_session_id=replay.source_session_id,
                runner=runner,
                omitted=replay.omitted,
            )
        )

    def action_rerun_history(self) -> None:
        """Return the selected prompt for a fresh recommendation path."""

        session = next(
            (item for item in self.sessions if item.id == self.selected_session_id),
            None,
        )
        if session is None:
            self.status_message = "No played or saved path is selected."
            self.refresh_details()
            return
        self.dismiss(
            HistoryRerunRequest(
                source_session_id=session.id,
                prompt=session.prompt,
            )
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Refresh the detail panel when the list cursor moves."""

        if event.data_table.id != "history-list" or event.row_key is None:
            return
        self.selected_session_id = int(str(event.row_key.value))
        self.status_message = ""
        self.refresh_details()

    def refresh_details(self) -> None:
        """Render the selected path, queue availability, and replay status."""

        details = self.query_one("#history-details", Static)
        if self.selected_session_id is None:
            details.update(
                Text.assemble(
                    ("No listening history yet.\n\n", "bold"),
                    ("Play a path or save one with `tonepath history save`.", "dim"),
                )
            )
            return
        try:
            record = load_history(self.store, self.selected_session_id)
        except RuntimeError as exc:
            details.update(Text(str(exc), style="bold red"))
            return
        details.update(_history_details(self.store, record, self.status_message))


def history_replay_status(store: TonepathStore, session: HistorySession) -> str:
    """Classify whether one history record can be replayed exactly."""

    if session.queue_count == 0:
        return "Legacy"
    record = load_history(store, session.id)
    available = sum(_queue_item_available(store, item) for item in record.queue)
    if available == 0:
        return "Unavailable"
    if available < len(record.queue):
        return "Partial"
    return "Ready"


def _history_details(store: TonepathStore, record: HistoryRecord, status_message: str) -> Text:
    """Build a readable history detail panel without mutating local state."""

    text = Text()
    session = record.session
    title = session.bookmark_name or f"Session {session.id}"
    text.append(f"{title}\n", style="bold")
    text.append(f"{session.prompt}\n\n")
    text.append("Transition\n", style="bold")
    text.append(f"{session.source_state} -> {session.target_state} · {session.duration_sec // 60}m\n")
    if record.phases:
        text.append(" -> ".join(phase.label for phase in record.phases))
        text.append("\n\n")
    text.append("Original queue\n", style="bold")
    if not record.queue:
        text.append("No queue snapshot was recorded for this session.\n", style="yellow")
    for item in record.queue:
        available = _queue_item_available(store, item)
        marker = "✓" if available else "!"
        title_text = item.title or item.path.name
        artist = item.artist or "Unknown artist"
        style = "dim" if available else "yellow"
        text.append(
            f"{marker} {item.position + 1}. {item.phase_label} · {title_text} - {artist}\n",
            style=style,
        )
    replay_status = history_replay_status(store, session)
    text.append("\nReplay status\n", style="bold")
    if replay_status == "Legacy":
        text.append("Legacy · Exact replay unavailable.\n", style="bold yellow")
        text.append(
            "This session was created before Tonepath saved exact queue snapshots.\n",
            style="yellow",
        )
        text.append("Press Enter to run this Request again as a new path.\n", style="bold")
        text.append("This uses the current selector; it is not the original queue.\n", style="dim")
        text.append("CLI\n", style="bold")
        text.append(f"uv run tonepath listen {shlex.quote(session.prompt)}\n", style="dim")
    elif replay_status == "Unavailable":
        text.append("Unavailable · No saved files remain playable.\n", style="bold yellow")
        text.append("Press Enter to run this Request again as a new path.", style="bold")
    elif replay_status == "Partial":
        missing = sum(not _queue_item_available(store, item) for item in record.queue)
        text.append(f"Partial · Ready with {missing} missing file(s) omitted.\n", style="yellow")
        text.append("Enter loads the saved queue; r runs a new selection.", style="dim")
    else:
        text.append("Ready for exact replay. Enter loads it without autoplay.\n", style="green")
        text.append("Press r to run the Request again with the current selector.", style="dim")
    if status_message:
        text.append("\nStatus\n", style="bold yellow")
        text.append(status_message, style="yellow")
    return text


def _queue_item_available(store: TonepathStore, item: HistoryQueueItem) -> bool:
    """Return whether a history item resolves to a current or snapshot file."""

    current = store.get_track(item.track_id) if item.track_id is not None else None
    if current is not None and current.path.expanduser().exists():
        return True
    return item.path.expanduser().exists()


def _replay_status_text(status: str) -> Text:
    """Render one compact replay status with stable semantic color."""

    styles = {
        "Ready": "green",
        "Partial": "yellow",
        "Unavailable": "bold yellow",
        "Legacy": "dim yellow",
    }
    return Text(status, style=styles[status])


def _short_timestamp(value: str) -> str:
    """Return a compact local timestamp label from SQLite text."""

    return value.replace("T", " ")[:16]


def _truncate(value: str, limit: int) -> str:
    """Truncate one history label for a stable table width."""

    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"
