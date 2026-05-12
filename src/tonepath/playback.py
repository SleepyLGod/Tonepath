"""Local playback adapters."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


STOP_TIMEOUT_SEC = 5.0


class MpvAdapter:
    """Minimal local mpv playback adapter."""

    def available(self) -> bool:
        """Return whether mpv is available on PATH."""

        return shutil.which("mpv") is not None

    def build_command(self, paths: list[Path]) -> list[str]:
        """Build the mpv command for local files."""

        return ["mpv", "--no-terminal", "--force-window=no", "--audio-display=no", *[str(path) for path in paths]]

    def start(self, paths: list[Path]) -> subprocess.Popen[bytes]:
        """Start local playback with mpv and return the process."""

        if not self.available():
            raise RuntimeError("mpv is not installed or not available on PATH. Install mpv or run tonepath doctor.")
        return subprocess.Popen(
            self.build_command(paths),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_and_stop_on_interrupt(self, process: subprocess.Popen[bytes]) -> int:
        """Wait for playback, stopping mpv cleanly when the user interrupts."""

        try:
            return process.wait()
        except KeyboardInterrupt:
            self.stop_process(process)
            raise

    def play(self, paths: list[Path], dry_run: bool = False) -> list[str]:
        """Play local files with mpv or return the command in dry-run mode."""

        command = self.build_command(paths)
        if dry_run:
            return command
        process = self.start(paths)
        self.wait_and_stop_on_interrupt(process)
        return command

    def stop_process(self, process: subprocess.Popen[bytes]) -> None:
        """Terminate one mpv process, killing it if it does not exit quickly."""

        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STOP_TIMEOUT_SEC)

    def stop_pid(self, pid: int) -> bool:
        """Stop a recorded mpv process by PID."""

        try:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + STOP_TIMEOUT_SEC
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
                time.sleep(0.05)
            os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return False
