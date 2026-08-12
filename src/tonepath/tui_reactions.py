"""TUI browser for tracks hidden by an explicit dislike."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Header, Static

from tonepath.db import TonepathStore


class ReactionPreviewRequested(Message):
    """Ask the player to preview one disliked track."""

    def __init__(self, track_id: int) -> None:
        super().__init__()
        self.track_id = track_id


class ReactionPreviewStopRequested(Message):
    """Ask the player to stop an active disliked-track preview."""


class TrackReactionCleared(Message):
    """Notify the player that one disliked track was restored."""

    def __init__(self, track_id: int, label: str) -> None:
        super().__init__()
        self.track_id = track_id
        self.label = label


class DislikedTracksScreen(Screen[None]):
    """Small full-screen list for previewing and restoring hidden tracks."""

    CSS = """
    DislikedTracksScreen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    #disliked-heading {
        height: 3;
        padding: 1 2 0 2;
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #disliked-body {
        height: 1fr;
        padding: 1;
    }

    #disliked-list {
        width: 58%;
        min-width: 48;
        margin-right: 1;
        background: $panel;
        border: round $surface;
        border-title-color: $primary;
    }

    #disliked-list .datatable--header {
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #disliked-list .datatable--cursor {
        text-style: bold;
        color: $foreground;
        background: $primary;
    }

    #disliked-details {
        width: 42%;
        padding: 1 2;
        background: $panel;
        border: round $surface;
        border-title-color: $secondary;
    }

    #disliked-command-bar {
        height: 1;
        padding: 0 1;
        color: $foreground;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("h", "close_disliked", "Back", show=False, priority=True),
        Binding("escape", "close_disliked", "Back", show=False, priority=True),
        Binding("j", "next_track", "Next", show=False),
        Binding("k", "previous_track", "Previous", show=False),
        Binding("space", "preview_track", "Preview", show=False, priority=True),
        Binding("x", "stop_preview", "Stop preview", show=False, priority=True),
        Binding("enter", "restore_track", "Restore", show=False, priority=True),
    ]

    def __init__(self, store: TonepathStore) -> None:
        super().__init__()
        self.store = store
        self.rows: list[dict[str, object]] = []
        self.selected_track_id: int | None = None
        self.preview_track_id: int | None = None
        self.status_message = ""

    def compose(self) -> ComposeResult:
        """Compose the disliked-track browser."""

        yield Header()
        yield Static("Disliked Tracks", id="disliked-heading")
        with Horizontal(id="disliked-body"):
            yield DataTable(id="disliked-list")
            yield Static("", id="disliked-details")
        yield Static(
            " ↑/↓ or j/k  Browse   Space  Preview   x  Stop   Enter  Restore   Esc or h  Back ",
            id="disliked-command-bar",
        )

    def on_mount(self) -> None:
        """Load disliked tracks without changing current playback."""

        table = self.query_one("#disliked-list", DataTable)
        table.border_title = "Hidden from Future Requests"
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("ID", "Track", "Artist", "Updated")
        self.query_one("#disliked-details", Static).border_title = "Reaction"
        self.reload_rows()
        table.focus()

    def reload_rows(self) -> None:
        """Reload current disliked reactions and preserve a valid selection."""

        self.rows = self.store.list_track_reactions("disliked")
        table = self.query_one("#disliked-list", DataTable)
        table.clear(columns=False)
        for row in self.rows:
            table.add_row(
                str(row["track_id"]),
                _truncate(str(row["title"] or "unknown"), 34),
                _truncate(str(row["artist"] or "unknown"), 26),
                str(row["updated_at"]),
                key=str(row["track_id"]),
            )
        available_ids = [int(row["track_id"]) for row in self.rows]
        if self.selected_track_id not in available_ids:
            self.selected_track_id = available_ids[0] if available_ids else None
        if self.selected_track_id is not None:
            table.move_cursor(row=available_ids.index(self.selected_track_id))
        self.refresh_details()

    def action_close_disliked(self) -> None:
        """Stop a preview, if any, and return to the loaded path."""

        if self.preview_track_id is not None:
            self.post_message(ReactionPreviewStopRequested())
        self.dismiss(None)

    def action_next_track(self) -> None:
        """Move to the next disliked track."""

        self.query_one("#disliked-list", DataTable).action_cursor_down()

    def action_previous_track(self) -> None:
        """Move to the previous disliked track."""

        self.query_one("#disliked-list", DataTable).action_cursor_up()

    def action_preview_track(self) -> None:
        """Ask the app to preview the selected track."""

        if self.selected_track_id is None:
            self.status_message = "No disliked track is selected."
            self.refresh_details()
            return
        self.post_message(ReactionPreviewRequested(self.selected_track_id))

    def action_stop_preview(self) -> None:
        """Stop only the disliked-track preview."""

        if self.preview_track_id is None:
            self.status_message = "No disliked-track preview is playing."
            self.refresh_details()
            return
        self.post_message(ReactionPreviewStopRequested())

    def action_restore_track(self) -> None:
        """Clear the selected dislike without changing the loaded path."""

        if self.selected_track_id is None:
            self.status_message = "No disliked track is selected."
            self.refresh_details()
            return
        row = next((item for item in self.rows if int(item["track_id"]) == self.selected_track_id), None)
        if row is None:
            self.reload_rows()
            return
        track_id = self.selected_track_id
        label = str(row["title"] or "unknown")
        self.store.clear_track_reaction(track_id)
        self.status_message = f"Restored {label}; future Requests may use it again."
        self.post_message(TrackReactionCleared(track_id, label))
        self.reload_rows()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Refresh details for the highlighted reaction."""

        if event.data_table.id != "disliked-list" or event.row_key is None:
            return
        self.selected_track_id = int(str(event.row_key.value))
        self.status_message = ""
        self.refresh_details()

    def set_preview_status(self, track_id: int | None, message: str) -> None:
        """Update preview state after the app handles playback."""

        self.preview_track_id = track_id
        self.status_message = message
        self.refresh_details()

    def refresh_details(self) -> None:
        """Render the selected reaction and preview consequences."""

        details = self.query_one("#disliked-details", Static)
        if self.selected_track_id is None:
            details.update(
                Text.assemble(
                    ("No disliked tracks.\n\n", "bold"),
                    ("Press Esc to return. Future Requests may use the full library.", "dim"),
                )
            )
            return
        row = next((item for item in self.rows if int(item["track_id"]) == self.selected_track_id), None)
        if row is None:
            details.update("No disliked track is selected.")
            return
        lines = [
            str(row["title"] or "unknown"),
            str(row["artist"] or "unknown"),
            "",
            "This track is hidden from future Requests.",
            "Space previews it and stops current playback, but keeps the loaded path.",
            "Enter restores it to future recommendations.",
        ]
        if self.status_message:
            lines.extend(("", self.status_message))
        details.update("\n".join(lines))


def _truncate(value: str, limit: int) -> str:
    """Keep reaction rows stable in narrow terminals."""

    if len(value) <= limit:
        return value
    return f"{value[: max(limit - 1, 0)]}…"
