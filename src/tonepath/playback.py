"""Local playback adapters."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path


STOP_TIMEOUT_SEC = 5.0
IPC_READY_TIMEOUT_SEC = 2.0


class MpvCommandError(RuntimeError):
    """Report an error returned by mpv over its local control socket."""

    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__(f"mpv command failed: {error}")


class MpvAdapter:
    """Minimal local mpv playback adapter."""

    def available(self) -> bool:
        """Return whether mpv is available on PATH."""

        return shutil.which("mpv") is not None

    def build_command(
        self,
        paths: list[Path],
        ipc_path: Path | None = None,
        volume: float | None = None,
    ) -> list[str]:
        """Build the mpv command for local files."""

        command = ["mpv", "--no-terminal", "--force-window=no", "--audio-display=no"]
        if ipc_path is not None:
            command.append(f"--input-ipc-server={ipc_path}")
        if volume is not None:
            command.append(f"--volume={volume:g}")
        return [*command, *[str(path) for path in paths]]

    def start(
        self,
        paths: list[Path],
        ipc_path: Path | None = None,
        volume: float | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start local playback with mpv and return the process."""

        if not self.available():
            raise RuntimeError("mpv is not installed or not available on PATH. Install mpv or run tonepath doctor.")
        return subprocess.Popen(
            self.build_command(paths, ipc_path=ipc_path, volume=volume),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def send_command(self, ipc_path: Path, command: list[object]) -> object:
        """Send one JSON command to a local mpv IPC socket."""

        request_id = 1
        request = json.dumps(
            {"command": command, "request_id": request_id},
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(1.0)
                client.connect(str(ipc_path))
                client.sendall(request)
                buffered = b""
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    buffered += chunk
                    while b"\n" in buffered:
                        line, buffered = buffered.split(b"\n", 1)
                        if not line:
                            continue
                        payload = decode_ipc_response(line)
                        if payload.get("request_id") != request_id:
                            continue
                        error = payload.get("error")
                        if error != "success":
                            raise MpvCommandError(str(error or "unknown error"))
                        return payload.get("data")
        except OSError as exc:
            raise RuntimeError(f"Could not communicate with mpv: {exc}") from exc
        raise RuntimeError("mpv closed its local control socket without replying to the command.")

    def wait_for_ipc(
        self,
        ipc_path: Path,
        process: subprocess.Popen[bytes],
        timeout: float = IPC_READY_TIMEOUT_SEC,
    ) -> None:
        """Wait until a newly started mpv process accepts local commands."""

        deadline = time.monotonic() + timeout
        last_error: RuntimeError | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("mpv exited before its local control socket became ready.")
            try:
                self.send_command(ipc_path, ["get_property", "pause"])
                return
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.02)
        detail = f": {last_error}" if last_error is not None else ""
        raise RuntimeError(f"mpv local control socket did not become ready{detail}")

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


def decode_ipc_response(response: bytes) -> dict[str, object]:
    """Decode one newline-delimited mpv IPC message."""

    try:
        payload = json.loads(response)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("mpv returned invalid JSON over its local control socket.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("mpv returned an invalid response over its local control socket.")
    return payload
