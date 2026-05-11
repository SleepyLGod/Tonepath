"""Local playback adapters."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MpvAdapter:
    """Minimal local mpv playback adapter."""

    def available(self) -> bool:
        """Return whether mpv is available on PATH."""

        return shutil.which("mpv") is not None

    def play(self, paths: list[Path], dry_run: bool = False) -> list[str]:
        """Play local files with mpv or return the command in dry-run mode."""

        command = ["mpv", "--no-terminal", "--force-window=no", "--audio-display=no", *[str(path) for path in paths]]
        if dry_run:
            return command
        if not self.available():
            raise RuntimeError("mpv is not installed or not available on PATH. Install mpv or run tonepath doctor.")
        subprocess.Popen(command)
        return command

