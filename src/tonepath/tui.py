"""Textual terminal interface for Tonepath."""

from __future__ import annotations

import shutil
from typing import Any

from rich.text import Text

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.display import display_artist, fallback_track_label
from tonepath.experience import smart_plan_session
from tonepath.evaluation import evaluate_rerank
from tonepath.llm import provider_config
from tonepath.model_runtime import model_runtime_status
from tonepath.models import CandidateScore, FeedbackType
from tonepath.playback_controller import PlaybackController
from tonepath.profile import profile_learning_hint
from tonepath.readiness import LibraryStatus, library_status, readiness_blocks_session, readiness_label, status_next_action
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
PLAYBACK_MODES = ("Manual", "Continue Path", "Repeat One", "Repeat Path")


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
        border-title-color: #6fb7a6;
    }

    #event-log {
        height: 8;
        margin: 0 1;
        background: #151914;
        border: round #3a4038;
        border-title-color: #a7afa5;
        color: #a7afa5;
    }
    """

    BINDINGS = [
        Binding("/", "focus_prompt", "Prompt"),
        Binding("n", "new_prompt", "New", show=False),
        Binding("space", "play", "Play", key_display="Space"),
        Binding("p", "play", "Play", show=False),
        Binding("x", "stop_playback", "Stop"),
        Binding(">", "next_track", "Next", key_display=">"),
        Binding("<", "previous_track", "Prev", key_display="<"),
        Binding("s", "skip", "Skip"),
        Binding("l", "like", "Like"),
        Binding("m", "cycle_playback_mode", "Mode"),
        Binding("i", "ai_assist", "AI Assist", show=False),
        Binding("?", "toggle_help", "Help", key_display="?"),
        Binding("e", "toggle_events", "Events", show=False),
        Binding("v", "no_vocals", "No vocals", show=False),
        Binding("a", "codex_audit", "Audit", show=False),
        Binding("r", "codex_rerank", "Rerank", show=False),
        Binding("+", "too_loud", "Quieter", show=False),
        Binding("-", "too_slow", "More energy", show=False),
        Binding("w", "why", "Why", show=False),
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
        self.model_runtime_ready = False
        self.library_status: LibraryStatus | None = None
        self.readiness = "Needs setup"
        self.readiness_action = "Run `uv run tonepath setup --preset private`."
        self.intent_note: str | None = None
        self.playback_mode = "Manual"
        self.events_expanded = False
        self.right_panel = "why"

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
        yield RichLog(id="event-log", wrap=True, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """Load local state and render the first session view."""

        self.store = TonepathStore()
        self.apply_panel_titles()
        table = self.query_one("#queue", DataTable)
        table.add_columns("#", "Phase", "Track", "Fit", "Energy", "Conf")

        self.model_runtime_ready = model_runtime_status().ready
        self.refresh_readiness()
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

    def action_cycle_playback_mode(self) -> None:
        """Cycle between manual, path, and repeat playback modes."""

        index = PLAYBACK_MODES.index(self.playback_mode)
        self.playback_mode = PLAYBACK_MODES[(index + 1) % len(PLAYBACK_MODES)]
        self.log_event(f"Playback mode: {self.playback_mode}.")
        self.refresh_session_view()

    def action_toggle_help(self) -> None:
        """Toggle the right panel between explanation and key help."""

        self.right_panel = "why" if self.right_panel == "help" else "help"
        self.log_event("Showing help panel." if self.right_panel == "help" else "Showing why panel.")
        self.refresh_session_view()

    def action_ai_assist(self) -> None:
        """Show local AI Assist status without changing config or calling a model."""

        self.right_panel = "why" if self.right_panel == "ai_assist" else "ai_assist"
        self.log_event("Showing AI Assist status." if self.right_panel == "ai_assist" else "Showing why panel.")
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
        self.refresh_readiness()
        if readiness_blocks_session(self.readiness):
            self.runner = None
            self.playback_status = "Needs setup"
            self.render_intake()
            self.log_event(f"Not ready for recommendations: {self.readiness}.")
            self.log_event(self.readiness_action)
            self.query_one("#prompt-input", Input).focus()
            return
        if self.playback is not None:
            self.playback.stop_current()
        self.right_panel = "why"
        settings = config.load_config()
        plan, note = smart_plan_session(cleaned, settings)
        self.intent_note = note
        self.runner = SessionRunner(self.store, cleaned, plan=plan)
        self.playback = self.playback or PlaybackController(self.store)
        self.playback_status = "Ready"
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = cleaned
        prompt_input.blur()
        if note:
            self.log_event(note)
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
        self.log_event(f"Playing: {fallback_track_label(candidate.track.title, candidate.track.path.name)}")
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
        self.log_event("Playback finished.")
        self.refresh_session_view()

    def refresh_session_view(self) -> None:
        """Refresh timeline, queue, why panel, and playback status."""

        if self.runner is None:
            self.render_intake()
            return

        self.apply_panel_titles()
        self.query_one("#status-bar", Static).update(self.status_bar_text())
        self.query_one("#timeline", Static).update(self.timeline_text())
        self.query_one("#now-playing", Static).update(self.now_playing_renderable())
        self.query_one("#why-panel", Static).update(self.why_panel_renderable())
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
                queue_cell(truncate(fallback_track_label(candidate.track.title, candidate.track.path.name), 28), current=current),
                queue_cell(self.fit_label(candidate), current=current),
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
        meter = energy_meter(features.energy if features is not None else None)
        loudness = "--" if features is None or features.loudness is None else f"{features.loudness:.1f} dBFS"
        bpm = bpm_text(features.bpm if features is not None else None)
        return "\n".join(
            [
                f"{self.playback_status} · {candidate.phase.label} · {self.playback_mode} · {confidence_label(candidate.confidence)}",
                truncate(fallback_track_label(candidate.track.title, candidate.track.path.name), 44),
                display_artist(candidate.track),
                f"energy {energy} {meter} · bpm {bpm}",
                f"loudness {loudness}",
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
            elif "▮" in line or "▯" in line:
                text.append(line, style=AMBER)
            else:
                text.append(line, style=AMBER_DIM)
        return text

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
                "Space / p  play current track",
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
                "i          AI Assist status",
                "e          expand/collapse events",
                "w          write full why to events",
                "a / r      Codex audit / rerank preview",
                "/ / n      prompt / new request",
                "q          quit",
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
            if line in {"Why", "Evidence", "Help", "Playback", "Feedback", "Tools", "AI Assist"} or line.startswith("Missing evidence"):
                if text:
                    text.append("\n")
                style = TEAL if line == "Evidence" else AMBER if line in {"Why", "Help", "Playback", "Feedback", "Tools", "AI Assist"} else MUTED
                text.append(line, style=f"bold {style}")
                continue
            text.append("\n")
            style = MUTED if line.startswith("Missing evidence") else TEXT
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
        self.query_one("#status-bar", Static).update(self.status_bar_text(extra="Enter to plan"))
        self.query_one("#timeline", Static).update("No session yet · feeling → path → feedback → memory")
        self.query_one("#now-playing", Static).update(
            Text.assemble(
                ("● No session yet\n", f"bold {AMBER}"),
                (f"{guidance}\n", TEXT),
                ("Example:\n", MUTED),
                (PROMPT_PLACEHOLDER, AMBER_DIM),
            )
        )
        self.query_one("#why-panel", Static).update(
            self.why_panel_renderable()
        )
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
            f"● {self.playback_status}",
            self.playback_mode,
            self.experience_label(),
            model_state,
            llm_state,
            f"{self.library_count()} tracks",
        ]
        if extra:
            parts.append(extra)
        return "   " + " · ".join(parts)

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
            "No scanned tracks.\n\nRun:\nuv run tonepath setup --preset private\nuv run tonepath config add-music-dir /path/to/music\nuv run tonepath prepare"
        )
        self.query_one("#why-panel", Static).update("Why panel appears after a local session starts.")
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
        self.query_one("#event-log", RichLog).border_title = "Events expanded" if self.events_expanded else "Events"
        self.query_one("#prompt-input", Input).border_title = "Request"


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


def queue_cell(value: str, current: bool = False, align: str | None = None) -> Text:
    """Return a styled queue table cell."""

    style = f"bold {AMBER}" if current else MUTED
    return Text(value, style=style, justify=align)


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
