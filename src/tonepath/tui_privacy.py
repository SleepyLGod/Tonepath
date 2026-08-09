"""TUI Data & Privacy browser built on the shared privacy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.worker import Worker, WorkerState
from textual.widgets import DataTable, Header, Input, Static

from tonepath.privacy import (
    ALL_PERSONAL_CATEGORIES,
    PrivacyCategoryReport,
    PrivacyDeletePlan,
    PrivacyDeleteResult,
    PrivacyInventory,
    build_privacy_inventory,
    execute_privacy_delete,
    export_personal_data,
    plan_privacy_delete,
    render_delete_plan,
)


ALL_PERSONAL_ROW_ID = "all-personal"


@dataclass(frozen=True)
class PrivacyExportTaskResult:
    """Successful background export result."""

    output: Path


@dataclass(frozen=True)
class PrivacyDeleteTaskResult:
    """Background delete result or stale-plan refusal."""

    result: PrivacyDeleteResult | None
    stale_message: str | None = None


class PrivacyDataDeleted(Message):
    """Notify the player that active local data was deleted."""

    def __init__(self, result: PrivacyDeleteResult) -> None:
        super().__init__()
        self.result = result


class PrivacyDeleteModal(ModalScreen[PrivacyDeletePlan | None]):
    """Typed confirmation for one immutable privacy deletion plan."""

    CSS = """
    PrivacyDeleteModal {
        align: center middle;
        background: $background 70%;
    }

    #privacy-confirm-dialog {
        width: 82%;
        height: 82%;
        padding: 1 2;
        background: $panel;
        border: round $warning;
    }

    #privacy-confirm-heading {
        height: 2;
        text-style: bold;
        color: $warning;
    }

    #privacy-confirm-plan {
        height: 1fr;
        overflow-y: auto;
        color: $foreground;
    }

    #privacy-confirm-input {
        height: 3;
        margin-top: 1;
        border: round $warning;
        background: $surface;
    }

    #privacy-confirm-status {
        height: 2;
        color: $warning;
    }

    #privacy-confirm-command {
        height: 1;
        color: $foreground;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel_delete", "Cancel", show=False, priority=True),
    ]

    def __init__(self, plan: PrivacyDeletePlan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        """Compose the deletion plan and exact confirmation input."""

        with Vertical(id="privacy-confirm-dialog"):
            yield Static("Permanent deletion preview", id="privacy-confirm-heading")
            yield Static(render_delete_plan(self.plan), id="privacy-confirm-plan")
            yield Input(placeholder="Type lowercase delete", id="privacy-confirm-input")
            yield Static("", id="privacy-confirm-status")
            yield Static(" Enter  Confirm deletion    Esc  Cancel ", id="privacy-confirm-command")

    def on_mount(self) -> None:
        """Focus the typed confirmation input."""

        self.query_one("#privacy-confirm-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Accept only the exact lowercase confirmation word."""

        if event.input.id != "privacy-confirm-input":
            return
        if event.value != "delete":
            self.query_one("#privacy-confirm-status", Static).update(
                "Confirmation must be exactly lowercase delete. Nothing was deleted."
            )
            return
        self.dismiss(self.plan)

    def action_cancel_delete(self) -> None:
        """Close without deleting anything."""

        self.dismiss(None)


class PrivacyScreen(Screen[None]):
    """Full-screen local data inventory, export, and deletion surface."""

    CSS = """
    PrivacyScreen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    #privacy-heading {
        height: 3;
        padding: 1 2 0 2;
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #privacy-body {
        height: 1fr;
        padding: 1;
    }

    #privacy-list {
        width: 48%;
        min-width: 48;
        margin-right: 1;
        background: $panel;
        border: round $surface;
        border-title-color: $primary;
    }

    #privacy-list .datatable--header {
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #privacy-list .datatable--cursor {
        text-style: bold;
        color: $foreground;
        background: $primary;
    }

    #privacy-details {
        width: 52%;
        padding: 1 2;
        background: $panel;
        border: round $surface;
        border-title-color: $secondary;
        overflow-y: auto;
    }

    #privacy-command-bar {
        height: 1;
        padding: 0 1;
        color: $foreground;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "close_privacy", "Back", show=False, priority=True),
        Binding("j", "next_category", "Next", show=False),
        Binding("k", "previous_category", "Previous", show=False),
        Binding("e", "export_privacy", "Export", show=False, priority=True),
        Binding("d", "delete_privacy", "Delete", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.inventory: PrivacyInventory | None = None
        self.selected_category_id = "memory"
        self.busy: str | None = None
        self.status_message = "Reading local data inventory in background..."
        self._inventory_completion_message: str | None = None
        self.last_delete_result: PrivacyDeleteResult | None = None

    def compose(self) -> ComposeResult:
        """Compose the category browser using the active Tonepath theme."""

        yield Header()
        yield Static("Data & Privacy · local, inspectable, and under your control", id="privacy-heading")
        with Horizontal(id="privacy-body"):
            yield DataTable(id="privacy-list")
            yield Static("", id="privacy-details")
        yield Static(
            " ↑/↓ or j/k  Browse   e  Export   d  Delete preview   Esc  Back ",
            id="privacy-command-bar",
        )

    def on_mount(self) -> None:
        """Start read-only inventory work without blocking the player."""

        table = self.query_one("#privacy-list", DataTable)
        table.border_title = "Local Data"
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Data", "Sensitivity", "Records", "Files", "Size", "Control")
        details = self.query_one("#privacy-details", Static)
        details.border_title = "What This Means"
        self.refresh_details()
        self.start_inventory_worker()

    def action_close_privacy(self) -> None:
        """Return to the player without changing its path."""

        if self.busy == "delete":
            self.status_message = "Confirmed deletion is still running. Wait for its result before leaving this screen."
            self.refresh_details()
            return
        self.dismiss(None)

    def action_next_category(self) -> None:
        """Move to the next privacy category."""

        self.query_one("#privacy-list", DataTable).action_cursor_down()

    def action_previous_category(self) -> None:
        """Move to the previous privacy category."""

        self.query_one("#privacy-list", DataTable).action_cursor_up()

    def action_export_privacy(self) -> None:
        """Export personal data in a background worker."""

        if self._show_busy_if_needed():
            return
        output = default_privacy_export_path()
        self.busy = "export"
        self.status_message = f"Exporting personal data in background to {output}... playback continues."
        self.refresh_details()
        self.run_worker(
            partial(self.export_job, output),
            name="privacy-export",
            group="privacy-export",
            description=self.status_message,
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def action_delete_privacy(self) -> None:
        """Build a background deletion preview for the selected category."""

        if self._show_busy_if_needed():
            return
        if self.selected_category_id in {"library-evidence", "models-storage"}:
            self.status_message = "This category is read-only in Privacy Center v1. Nothing was deleted."
            self.refresh_details()
            return
        categories = (
            ALL_PERSONAL_CATEGORIES
            if self.selected_category_id == ALL_PERSONAL_ROW_ID
            else (self.selected_category_id,)
        )
        blocker = deletion_task_blocker(self.app, categories)
        if blocker:
            self.status_message = blocker
            self.refresh_details()
            return
        self.busy = "plan"
        self.status_message = "Preparing an exact deletion preview in background... playback continues."
        self.refresh_details()
        self.run_worker(
            partial(plan_privacy_delete, categories),
            name="privacy-plan",
            group="privacy-plan",
            description=self.status_message,
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def on_delete_confirmed(self, plan: PrivacyDeletePlan | None) -> None:
        """Execute one confirmed plan in a background worker."""

        if plan is None:
            self.status_message = "Deletion cancelled. Nothing was deleted."
            self.refresh_details()
            return
        blocker = deletion_task_blocker(self.app, plan.categories)
        if blocker:
            self.status_message = blocker
            self.refresh_details()
            return
        self.busy = "delete"
        self.status_message = "Deleting confirmed personal data in background... playback continues until the result is known."
        self.refresh_details()
        self.run_worker(
            partial(self.delete_job, plan),
            name="privacy-delete",
            group="privacy-delete",
            description=self.status_message,
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def start_inventory_worker(self, completion_message: str | None = None) -> None:
        """Refresh inventory off-thread, optionally preserving an operation result."""

        self.busy = "inventory"
        self._inventory_completion_message = completion_message
        if completion_message is None:
            self.status_message = "Reading local data inventory in background..."
        self.refresh_details()
        self.run_worker(
            build_privacy_inventory,
            name="privacy-inventory",
            group="privacy-inventory",
            description="Reading local privacy inventory",
            exit_on_error=False,
            exclusive=True,
            thread=True,
        )

    def export_job(self, output: Path) -> PrivacyExportTaskResult:
        """Write a sanitized personal-data bundle off the TUI thread."""

        return PrivacyExportTaskResult(export_personal_data(output))

    def delete_job(self, plan: PrivacyDeletePlan) -> PrivacyDeleteTaskResult:
        """Execute a confirmed plan, returning stale-plan errors as normal state."""

        try:
            return PrivacyDeleteTaskResult(execute_privacy_delete(plan))
        except RuntimeError as exc:
            if "changed since the preview" in str(exc):
                return PrivacyDeleteTaskResult(
                    result=None,
                    stale_message=f"{exc} Review the refreshed inventory and confirm again.",
                )
            raise

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Apply completed privacy work on the Textual event thread."""

        worker = event.worker
        if not worker.group.startswith("privacy-"):
            return
        if event.state not in {WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED}:
            return
        self.busy = None
        if event.state == WorkerState.ERROR:
            self.status_message = f"Privacy task failed: {worker.error}. Nothing else was changed by this task."
            self.refresh_details()
            return
        if event.state == WorkerState.CANCELLED:
            self.status_message = "Privacy task cancelled."
            self.refresh_details()
            return
        result = worker.result
        if worker.group == "privacy-inventory" and isinstance(result, PrivacyInventory):
            self.inventory = result
            self.refresh_table()
            self.status_message = self._inventory_completion_message or "Inventory ready. Browsing does not change playback or the current path."
            self._inventory_completion_message = None
            self.refresh_details()
            return
        if worker.group == "privacy-plan" and isinstance(result, PrivacyDeletePlan):
            self.status_message = "Preview ready. Nothing is deleted until lowercase delete is entered."
            self.refresh_details()
            self.app.push_screen(PrivacyDeleteModal(result), self.on_delete_confirmed)
            return
        if worker.group == "privacy-export" and isinstance(result, PrivacyExportTaskResult):
            self.status_message = f"Personal data exported to {result.output}. Current playback and queue are unchanged."
            self.refresh_details()
            return
        if worker.group == "privacy-delete" and isinstance(result, PrivacyDeleteTaskResult):
            if result.stale_message:
                self.start_inventory_worker(completion_message=result.stale_message)
                return
            if result.result is None:
                self.status_message = "Deletion returned no result. Nothing else was changed."
                self.refresh_details()
                return
            delete_result = result.result
            self.last_delete_result = delete_result
            self.post_message(PrivacyDataDeleted(delete_result))
            changed = ", ".join(delete_result.changed_categories) or "none"
            if delete_result.failed:
                message = (
                    f"Deletion partially completed. Changed categories: {changed}. "
                    "Review failed items and rerun the same preview."
                )
            else:
                message = f"Deletion complete. Changed categories: {changed}."
            self.start_inventory_worker(completion_message=message)
            return
        self.status_message = "Privacy task finished with an unreadable result."
        self.refresh_details()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Refresh category details when the list cursor moves."""

        if event.data_table.id != "privacy-list" or event.row_key is None:
            return
        self.selected_category_id = str(event.row_key.value)
        self.refresh_details()

    def refresh_table(self) -> None:
        """Render the current five-category inventory plus All Personal Data."""

        table = self.query_one("#privacy-list", DataTable)
        table.clear(columns=False)
        if self.inventory is None:
            return
        for category in self.inventory.categories:
            control = "Delete" if "delete" in category.capabilities else "Read-only"
            table.add_row(
                category.label,
                category.sensitivity.title(),
                str(category.record_count),
                str(category.file_count),
                human_bytes(category.file_size_bytes),
                control,
                key=category.id,
            )
        personal = [
            category
            for category in self.inventory.categories
            if category.id in ALL_PERSONAL_CATEGORIES
        ]
        table.add_row(
            "All Personal Data",
            "High",
            str(sum(category.record_count for category in personal)),
            str(sum(category.file_count for category in personal)),
            human_bytes(sum(category.file_size_bytes for category in personal)),
            "Delete",
            key=ALL_PERSONAL_ROW_ID,
        )
        row_keys = [str(key.value) for key in table.rows]
        if self.selected_category_id not in row_keys:
            self.selected_category_id = "memory"
        table.move_cursor(row=row_keys.index(self.selected_category_id))

    def refresh_details(self) -> None:
        """Render user-facing category meaning, controls, and task status."""

        details = self.query_one("#privacy-details", Static)
        text = Text()
        if self.status_message:
            text.append("Status\n", style="bold yellow" if self.busy else "bold")
            text.append(f"{self.status_message}\n\n", style="yellow" if self.busy else "")
        if self.inventory is None:
            text.append("Inventory is loading. Playback continues in the background.", style="dim")
            details.update(text)
            return
        if self.selected_category_id == ALL_PERSONAL_ROW_ID:
            text.append("All Personal Data\n", style="bold")
            text.append("Memory + Personalization + Listening History\n\n")
            text.append("Delete effect\n", style="bold")
            text.append("Removes private notes, learned preferences, and recorded listening paths.\n")
            text.append("Keeps\n", style="bold")
            text.append("Configuration, original music, library evidence, models, runtimes, and non-personal caches.\n")
        else:
            category = self._selected_category()
            if category is None:
                details.update(text)
                return
            text.append(f"{category.label}\n", style="bold")
            text.append(f"{category.description}\n\n")
            text.append("Stored locally\n", style="bold")
            text.append(
                f"{category.record_count} database rows · {category.file_count} files · "
                f"{human_bytes(category.file_size_bytes)}\n"
            )
            if category.records:
                text.append(" · ".join(f"{name} {count}" for name, count in category.records.items()))
                text.append("\n")
            text.append("\nControls\n", style="bold")
            text.append(" · ".join(category.capabilities))
            text.append("\n")
            for effect in category.effects:
                text.append(f"{effect}\n", style="dim")
        external = self.inventory.external_processing
        text.append("\nExternal Processing\n", style="bold")
        text.append(
            f"Policy {'allowed' if external['allowed'] else 'off'} · "
            f"Provider {external['provider']} · "
            f"Key {'present' if external['key_present'] else 'missing'}\n"
        )
        text.append("Transmission history: not recorded", style="dim")
        if self.last_delete_result is not None:
            text.append("\n\nLast deletion result\n", style="bold")
            for item in self.last_delete_result.items:
                style = "red" if item.status == "failed" else "dim"
                text.append(
                    f"{item.status.upper()} · {item.component}: {item.message}\n",
                    style=style,
                )
        details.update(text)

    def _selected_category(self) -> PrivacyCategoryReport | None:
        if self.inventory is None:
            return None
        return next(
            (category for category in self.inventory.categories if category.id == self.selected_category_id),
            None,
        )

    def _show_busy_if_needed(self) -> bool:
        if self.busy is None:
            return False
        self.status_message = f"Privacy task already running: {self.busy}. Playback continues."
        self.refresh_details()
        return True


def deletion_task_blocker(app: Any, categories: tuple[str, ...]) -> str | None:
    """Refuse deletion while an active worker could recreate the same data."""

    if any(category in {"memory", "personalization"} for category in categories) and bool(
        getattr(app, "memory_busy", False)
    ):
        return "Memory learning is still running. Wait for it to finish before deleting Memory or Personalization."
    if "history" in categories and bool(getattr(app, "request_busy", False)):
        return "Request planning is still running. Wait for it to finish before deleting Listening History."
    return None


def database_records_cleared(result: PrivacyDeleteResult, category: str) -> bool:
    """Return whether one category's SQLite component is deleted or already absent."""

    component = f"SQLite {category} records"
    return any(
        item.component == component and item.status in {"deleted", "already_absent"}
        for item in result.items
    )


def category_delete_completed(result: PrivacyDeleteResult, category: str) -> bool:
    """Return whether every attempted component for one category completed."""

    items = [item for item in result.items if item.category == category]
    return bool(items) and all(item.status in {"deleted", "already_absent"} for item in items)


def default_privacy_export_path() -> Path:
    """Return the default timestamped owner export directory."""

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path.home() / f"tonepath-personal-data-{timestamp}"


def human_bytes(value: int) -> str:
    """Return a compact binary size label for the privacy table."""

    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"
