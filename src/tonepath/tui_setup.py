"""TUI-only guided setup screen built on the shared setup draft."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Header, Input, Static

from tonepath import config
from tonepath.llm import provider_config
from tonepath.setup import SetupDraft, setup_review, validate_music_directories


@dataclass(frozen=True)
class SetupOutcome:
    """Confirmed setup choices returned to the player app."""

    settings: config.TonepathConfig
    prepare_requested: bool
    setup_models: bool


class SetupScreen(Screen[SetupOutcome | None]):
    """Full-screen first-run and selective reconfiguration workflow."""

    CSS = """
    SetupScreen {
        layout: vertical;
        background: $background;
        color: $foreground;
    }

    #setup-heading {
        height: 3;
        padding: 1 2 0 2;
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #setup-body {
        height: 1fr;
        padding: 1;
    }

    #setup-options {
        width: 46%;
        min-width: 44;
        margin-right: 1;
        background: $panel;
        border: round $surface;
        border-title-color: $primary;
    }

    #setup-options .datatable--header {
        text-style: bold;
        color: $primary;
        background: $surface;
    }

    #setup-options .datatable--cursor {
        text-style: bold;
        color: $foreground;
        background: $primary;
    }

    #setup-details {
        width: 54%;
        padding: 1 2;
        background: $panel;
        border: round $surface;
        border-title-color: $secondary;
        overflow-y: auto;
    }

    #setup-music-input {
        height: 3;
        margin: 0 1;
        background: $panel;
        border: round $primary;
    }

    #setup-status {
        height: 2;
        padding: 0 2;
        color: $warning;
        background: $background;
    }

    #setup-command-bar {
        height: 1;
        padding: 0 1;
        color: $foreground;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back", show=False, priority=True),
        Binding("j", "next_option", "Next", show=False),
        Binding("k", "previous_option", "Previous", show=False),
        Binding("enter", "select_option", "Select", show=False),
    ]

    def __init__(
        self,
        settings: config.TonepathConfig,
        *,
        first_run: bool,
        model_ready: bool,
    ) -> None:
        super().__init__()
        self.base_settings = settings
        self.draft = SetupDraft.from_config(settings)
        self.first_run = first_run
        self.model_ready = model_ready
        self.state = "music-input" if first_run else "summary"
        self.selected_option: str | None = None
        self.status_message = ""
        self._return_state = "review" if first_run else "summary"

    def compose(self) -> ComposeResult:
        """Compose the setup browser and one explicit music-path input."""

        yield Header()
        yield Static("", id="setup-heading")
        with Horizontal(id="setup-body"):
            yield DataTable(id="setup-options")
            yield Static("", id="setup-details")
        yield Input(placeholder="Local music directory", id="setup-music-input")
        yield Static("", id="setup-status")
        yield Static(" ↑/↓ or j/k  Choose   Enter  Continue   Esc  Back ", id="setup-command-bar")

    def on_mount(self) -> None:
        """Initialize stable widgets and show the first setup state."""

        table = self.query_one("#setup-options", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Choice", "What it means")
        self.query_one("#setup-details", Static).border_title = "What This Changes"
        self.show_state(self.state)

    def action_next_option(self) -> None:
        """Move to the next setup choice."""

        self.query_one("#setup-options", DataTable).action_cursor_down()

    def action_previous_option(self) -> None:
        """Move to the previous setup choice."""

        self.query_one("#setup-options", DataTable).action_cursor_up()

    def action_select_option(self) -> None:
        """Apply the highlighted choice and advance the setup state."""

        if self.state == "music-input":
            return
        choice = self.selected_option
        if choice is None:
            return
        self.status_message = ""
        if self.state == "summary":
            self.select_summary(choice)
        elif self.state == "music-menu":
            self.select_music_menu(choice)
        elif self.state == "experience":
            self.select_experience(choice)
        elif self.state == "ai-consent":
            self.select_ai_consent(choice)
        elif self.state == "models":
            self.draft = self.draft.with_models(choice, allow_setup=self.draft.allow_model_setup)
            self.finish_section()
        elif self.state == "local-data":
            self.draft = self.draft.with_local_history(choice == "store")
            self.finish_section()
        elif self.state == "advanced-model":
            self.draft = self.draft.with_models(choice, allow_setup=self.draft.allow_model_setup)
            self.show_state("advanced-provider")
        elif self.state == "advanced-provider":
            self.draft = self.draft.with_experience(
                "custom",
                send_to_llm=self.draft.send_to_llm,
                provider=choice,
            )
            self.show_state("ai-consent")
        elif self.state == "review":
            if choice == "back":
                self.show_state("experience" if self.first_run else "summary")
            else:
                self.show_state("prepare-choice")
        elif self.state == "prepare-choice":
            if choice == "later":
                self.finish_setup(prepare=False, setup_models=False)
            elif self.model_ready:
                self.finish_setup(prepare=True, setup_models=False)
            else:
                self.show_state("model-choice")
        elif self.state == "model-choice":
            self.finish_setup(prepare=True, setup_models=choice == "setup-models")

    def select_summary(self, choice: str) -> None:
        """Open one selective reconfiguration section."""

        if choice == "cancel":
            self.dismiss(None)
        elif choice == "review":
            self.show_state("review")
        elif choice == "prepare":
            self.show_state("review")
        elif choice == "music":
            self.show_state("music-menu")
        elif choice == "experience":
            self.show_state("experience")
        elif choice == "models":
            self.show_state("models")
        elif choice == "local-data":
            self.show_state("local-data")

    def select_music_menu(self, choice: str) -> None:
        """Keep, add, or explicitly remove one music directory."""

        if choice == "keep":
            self.show_state("summary")
            return
        if choice == "add":
            self._return_state = "summary"
            self.show_state("music-input")
            return
        if not choice.startswith("remove:"):
            return
        path = choice.removeprefix("remove:")
        updated = self.draft.remove_music_dir(Path(path))
        if not updated.music_dirs:
            self.status_message = "At least one music directory is required. Add another directory before removing this one."
            self.refresh_details()
            return
        self.draft = updated
        self.show_state("music-menu")

    def select_experience(self, choice: str) -> None:
        """Apply Private immediately or continue Smart/Custom substeps."""

        if choice == "private":
            self.draft = self.draft.with_experience("private", send_to_llm=False, provider=self.draft.llm_provider)
            self.finish_section()
        elif choice == "smart":
            self.draft = self.draft.with_experience("smart", send_to_llm=False, provider=self.draft.llm_provider)
            self.show_state("ai-consent")
        else:
            self.draft = self.draft.with_experience(
                "custom",
                send_to_llm=self.draft.send_to_llm,
                provider=self.draft.llm_provider,
            )
            self.show_state("advanced-model")

    def select_ai_consent(self, choice: str) -> None:
        """Store an explicit external-text consent choice."""

        self.draft = self.draft.with_experience(
            self.draft.experience_mode,
            send_to_llm=choice == "allow-ai",
            provider=self.draft.llm_provider,
        )
        if self.draft.experience_mode == "custom":
            self.show_state("local-data")
        else:
            self.finish_section()

    def finish_section(self) -> None:
        """Return from one setup section to Review or the current setup summary."""

        self.show_state("review" if self.first_run else "summary")

    def finish_setup(self, *, prepare: bool, setup_models: bool) -> None:
        """Dismiss with final settings only after all confirmations are complete."""

        try:
            validate_music_directories(self.draft.music_dirs)
        except ValueError as exc:
            self.show_state("music-input")
            self.status_message = str(exc)
            self.refresh_details()
            return
        self.dismiss(
            SetupOutcome(
                settings=self.draft.to_config(self.base_settings),
                prepare_requested=prepare,
                setup_models=setup_models,
            )
        )

    def action_back(self) -> None:
        """Return one level or leave setup without saving."""

        if self.state in {"summary", "music-input"} and self.first_run:
            self.dismiss(None)
        elif self.state == "summary":
            self.dismiss(None)
        elif self.state in {"review", "prepare-choice", "model-choice"}:
            self.show_state("experience" if self.first_run else "summary")
        elif self.first_run:
            self.show_state("music-input")
        else:
            self.show_state("summary")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Validate a local music directory before changing the setup draft."""

        if event.input.id != "setup-music-input" or self.state != "music-input":
            return
        raw_path = event.value.strip()
        try:
            validate_music_directories((raw_path,))
        except ValueError as exc:
            self.status_message = str(exc)
            self.refresh_details()
            event.input.focus()
            event.input.action_select_all()
            return
        if self.first_run and self._return_state == "review":
            self.draft = self.draft.replace_music_dirs((raw_path,))
            self.show_state("experience")
        else:
            self.draft = self.draft.add_music_dir(Path(raw_path))
            self.show_state(self._return_state)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the highlighted choice and refresh its explanation."""

        if event.data_table.id != "setup-options" or event.row_key is None:
            return
        self.selected_option = str(event.row_key.value)
        self.refresh_details()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Advance when the user presses Enter on the highlighted setup choice."""

        if event.data_table.id != "setup-options":
            return
        self.action_select_option()

    def show_state(self, state: str) -> None:
        """Render one stable page of the setup workflow."""

        self.state = state
        self.status_message = ""
        heading = "Getting Started" if self.first_run else "Setup"
        step = self.state_label()
        self.query_one("#setup-heading", Static).update(f"{heading} · {step}")
        input_widget = self.query_one("#setup-music-input", Input)
        table = self.query_one("#setup-options", DataTable)
        input_widget.display = state == "music-input"
        table.display = state != "music-input"
        if state == "music-input":
            input_widget.value = ""
            input_widget.border_title = "Music Directory"
            input_widget.focus()
            self.selected_option = None
        else:
            self.populate_options()
            table.focus()
        self.refresh_details()

    def populate_options(self) -> None:
        """Fill the option table for the active non-input state."""

        table = self.query_one("#setup-options", DataTable)
        table.clear(columns=False)
        options, title = self.state_options()
        table.border_title = title
        for key, label, description in options:
            table.add_row(label, description, key=key)
        if options:
            preferred = self.preferred_option_key()
            row = next((index for index, option in enumerate(options) if option[0] == preferred), 0)
            self.selected_option = options[row][0]
            table.move_cursor(row=row)

    def preferred_option_key(self) -> str | None:
        """Return the current saved choice for selective reconfiguration screens."""

        if self.state == "experience":
            return self.draft.experience_mode
        if self.state == "ai-consent":
            return "allow-ai" if self.draft.send_to_llm else "keep-local"
        if self.state in {"models", "advanced-model"}:
            return self.draft.model_mode
        if self.state == "advanced-provider":
            return self.draft.llm_provider
        if self.state == "local-data":
            return "store" if self.draft.store_play_history else "do-not-store"
        return None

    def state_options(self) -> tuple[list[tuple[str, str, str]], str]:
        """Return selectable rows and panel title for the active state."""

        if self.state == "summary":
            directory_count = len(self.draft.music_dirs)
            directory_label = "directory" if directory_count == 1 else "directories"
            return (
                [
                    ("music", "Music Library", f"{directory_count} local {directory_label}"),
                    ("experience", "Experience & AI", self.draft.experience_mode.title()),
                    ("models", "Local Models", self.draft.model_mode.title()),
                    ("local-data", "Local Data", "Playback history and text consent"),
                    ("prepare", "Prepare Library", "Scan and analyze after review"),
                    ("review", "Review & Save", "Save only after final confirmation"),
                    ("cancel", "Cancel", "Return without changing config"),
                ],
                "Current Setup",
            )
        if self.state == "music-menu":
            rows = [("keep", "Keep directories", "Return without changing the list"), ("add", "Add directory", "Preserve all existing directories")]
            rows.extend((f"remove:{path}", f"Remove {Path(path).name or path}", path) for path in self.draft.music_dirs)
            return rows, "Music Library"
        if self.state == "experience":
            return (
                [
                    ("private", "Private", "Local-only text processing; optional local models"),
                    ("smart", "Smart", "Local audio analysis plus separately consented AI Assist"),
                    ("custom", "Custom / Advanced", "Choose model mode, provider, consent, and local history"),
                ],
                "Experience",
            )
        if self.state == "ai-consent":
            return (
                [
                    ("keep-local", "Keep text local", "Use deterministic intent and local Memory files"),
                    ("allow-ai", "Allow AI Assist", "Send Request/Memory text only for opted-in AI tasks"),
                ],
                "External Text Processing",
            )
        if self.state in {"models", "advanced-model"}:
            return (
                [
                    ("fast", "Fast", "MIR only; no model-backed tags"),
                    ("balanced", "Balanced", "Use optional tags when the runtime is ready"),
                    ("full", "Full", "Expect local tags and affect evidence"),
                ],
                "Local Analysis",
            )
        if self.state == "advanced-provider":
            return (
                [
                    ("deepseek", "DeepSeek", "Reads DEEPSEEK_API_KEY; the key is never stored"),
                    ("qwen", "Qwen", "Reads QWEN_API_KEY; the key is never stored"),
                ],
                "AI Provider",
            )
        if self.state == "local-data":
            return (
                [
                    ("store", "Store playback history", "Keep local paths for replay and learning"),
                    ("do-not-store", "Do not store playback history", "Keep new playback history off"),
                ],
                "Local Data",
            )
        if self.state == "review":
            return (
                [
                    ("confirm", "Confirm setup", "Continue to the separate preparation choice"),
                    ("back", "Back", "Review or change setup choices"),
                ],
                "Review & Save",
            )
        if self.state == "prepare-choice":
            return (
                [
                    ("later", "Prepare later", "Save config now; run prepare when convenient"),
                    ("prepare", "Prepare library now", "Scan and run local analysis in the background"),
                ],
                "Prepare Library",
            )
        return (
            [
                ("base-only", "Base analysis only", "Run local MIR without downloading model runtimes"),
                ("setup-models", "Set up local models", "Download/setup the optional local tag and affect runtime"),
            ],
            "Optional Local Models",
        )

    def state_label(self) -> str:
        """Return a concise progress label without exposing internal state names."""

        if self.first_run:
            if self.state == "music-input":
                return "1/3 Music"
            if self.state in {"experience", "ai-consent", "advanced-model", "advanced-provider", "local-data"}:
                return "2/3 Experience"
            return "3/3 Review & Start"
        return "Current configuration"

    def refresh_details(self) -> None:
        """Explain the active step, current choices, and validation status."""

        details = self.query_one("#setup-details", Static)
        text = Text()
        if self.status_message:
            text.append("Needs attention\n", style="bold yellow")
            text.append(f"{self.status_message}\n\n", style="yellow")
        if self.state == "music-input":
            text.append("Music\n", style="bold")
            text.append("Enter one existing local folder. Tonepath scans audio from this folder but never uploads the files.\n")
        elif self.state == "review":
            text.append(setup_review(self.draft, model_ready=self.model_ready, provider_key_ready=self.provider_key_ready()))
        elif self.state == "summary":
            text.append(setup_review(self.draft, model_ready=self.model_ready, provider_key_ready=self.provider_key_ready()))
            text.append("\n\nChoose one area on the left. Unselected settings stay unchanged.", style="dim")
        elif self.state == "prepare-choice":
            text.append("Saving config and preparing music are separate decisions.\n\n")
            text.append("Prepare scans local files and writes local audio evidence. It does not change the current queue.", style="dim")
        elif self.state == "model-choice":
            text.append("Optional model setup is a separate network/download decision.\n\n")
            text.append("Base analysis still works when you choose not to install models.", style="dim")
        else:
            text.append(self.choice_explanation(), style="dim")
        details.update(text)
        self.query_one("#setup-status", Static).update(self.status_message)

    def choice_explanation(self) -> str:
        """Return the description for the currently highlighted option."""

        options, _ = self.state_options()
        for key, label, description in options:
            if key == self.selected_option:
                return f"{label}\n\n{description}"
        return "Choose an option to continue."

    def provider_key_ready(self) -> bool:
        """Return only whether the selected provider key exists."""

        try:
            return provider_config(self.draft.llm_provider).configured
        except ValueError:
            return False
