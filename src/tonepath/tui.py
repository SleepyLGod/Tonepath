"""Textual terminal interface for Tonepath."""

from __future__ import annotations


def run_tui() -> None:
    """Run a small TUI placeholder until the full interface is implemented."""

    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Footer, Header, Static
    except ImportError as exc:
        raise RuntimeError("Textual is not installed. Run `uv sync` before launching the TUI.") from exc

    class TonepathApp(App[None]):
        """Initial Tonepath terminal interface."""

        BINDINGS = [("q", "quit", "Quit")]

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static("Tonepath\n\nUse `tonepath start \"从烦躁到专注，30分钟\"` to start a session.")
            yield Static("v0 TUI shell: path timeline, queue, feedback, why panel, and privacy badge land here.")
            yield Footer()

    TonepathApp().run()

