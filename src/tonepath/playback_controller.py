"""Playback lifecycle management for Tonepath."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.playback import MpvAdapter, MpvCommandError


CURRENT_MPV_PID_KEY = "current_mpv_pid"
CURRENT_PLAY_ID_KEY = "current_play_id"
CURRENT_MPV_IPC_PATH_KEY = "current_mpv_ipc_path"


@dataclass(frozen=True)
class PlaybackState:
    """Live state reported by the managed mpv process."""

    playing: bool
    paused: bool
    position_sec: float | None
    duration_sec: float | None
    volume: float | None


class PlaybackController:
    """Own Tonepath-managed playback process state."""

    def __init__(self, store: TonepathStore, adapter: MpvAdapter | None = None) -> None:
        self.store = store
        self.adapter = adapter or MpvAdapter()
        self.process: subprocess.Popen[bytes] | None = None
        self.ipc_path: Path | None = None
        self.volume = 100.0

    def start(
        self,
        paths: list[Path],
        session_id: int | None = None,
        track_id: int | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start playback and store the managed process PID."""

        ipc_path = new_ipc_path()
        process = self.adapter.start(paths, ipc_path=ipc_path, volume=self.volume)
        try:
            self.adapter.wait_for_ipc(ipc_path, process)
        except RuntimeError:
            self.adapter.stop_process(process)
            self.process = None
            self.ipc_path = ipc_path
            self.clear()
            raise
        self.process = process
        self.ipc_path = ipc_path
        self.store.set_app_state(CURRENT_MPV_PID_KEY, str(process.pid))
        self.store.set_app_state(CURRENT_MPV_IPC_PATH_KEY, str(ipc_path))
        if track_id is not None:
            play_id = self.store.start_play(session_id=session_id, track_id=track_id)
            self.store.set_app_state(CURRENT_PLAY_ID_KEY, str(play_id))
        return process

    def state(self) -> PlaybackState:
        """Return live playback properties from mpv."""

        ipc_path = self.active_ipc_path()
        if ipc_path is None or (self.process is not None and self.process.poll() is not None):
            return PlaybackState(False, False, None, None, None)
        paused = bool(self.adapter.send_command(ipc_path, ["get_property", "pause"]))
        position = optional_float(read_optional_property(self.adapter, ipc_path, "time-pos"))
        duration = optional_float(read_optional_property(self.adapter, ipc_path, "duration"))
        volume = optional_float(read_optional_property(self.adapter, ipc_path, "volume"))
        if volume is not None:
            self.volume = volume
        return PlaybackState(True, paused, position, duration, volume)

    def pause(self) -> None:
        """Pause the active mpv process without ending its play record."""

        self.adapter.send_command(self.require_ipc_path(), ["set_property", "pause", True])

    def resume(self) -> None:
        """Resume the active mpv process without creating another play record."""

        self.adapter.send_command(self.require_ipc_path(), ["set_property", "pause", False])

    def toggle_pause(self) -> bool:
        """Toggle pause and return whether playback is now paused."""

        paused = self.state().paused
        self.adapter.send_command(self.require_ipc_path(), ["set_property", "pause", not paused])
        return not paused

    def seek_relative(self, seconds: float) -> None:
        """Seek relative to the current position without changing queue state."""

        self.adapter.send_command(self.require_ipc_path(), ["seek", seconds, "relative+exact"])

    def adjust_volume(self, delta: float) -> float:
        """Adjust managed mpv volume and return the clamped value."""

        self.volume = min(max(self.volume + delta, 0.0), 100.0)
        self.adapter.send_command(self.require_ipc_path(), ["set_property", "volume", self.volume])
        return self.volume

    def active_ipc_path(self) -> Path | None:
        """Return the active process IPC path, including recorded state."""

        if self.ipc_path is not None:
            return self.ipc_path
        value = self.store.get_app_state(CURRENT_MPV_IPC_PATH_KEY)
        return Path(value) if value else None

    def require_ipc_path(self) -> Path:
        """Return the active IPC path or fail with a user-facing error."""

        ipc_path = self.active_ipc_path()
        if ipc_path is None:
            raise RuntimeError("No controllable Tonepath playback is active.")
        return ipc_path

    def replace(
        self,
        paths: list[Path],
        session_id: int | None = None,
        track_id: int | None = None,
        mark_current_skipped: bool = False,
    ) -> subprocess.Popen[bytes]:
        """Stop current playback and start replacement playback."""

        self.stop_current(mark_skipped=mark_current_skipped)
        return self.start(paths, session_id=session_id, track_id=track_id)

    def wait_foreground(self, process: subprocess.Popen[bytes]) -> int:
        """Wait for foreground playback and clear PID state when it ends."""

        try:
            return self.adapter.wait_and_stop_on_interrupt(process)
        finally:
            self.clear()

    def stop_current(self, mark_skipped: bool = False) -> bool:
        """Stop the current managed process or recorded PID."""

        stopped = False
        if self.process is not None:
            self.adapter.stop_process(self.process)
            self.process = None
            stopped = True
        else:
            stopped = self.stop_recorded(mark_skipped=mark_skipped)
            return stopped
        self.clear(mark_skipped=mark_skipped)
        return stopped

    def stop_recorded(self, mark_skipped: bool = False) -> bool:
        """Stop the recorded Tonepath mpv PID without touching unrelated mpv processes."""

        pid = self.current_pid()
        if pid is None:
            self.clear()
            return False
        stopped = self.adapter.stop_pid(pid)
        self.clear(mark_skipped=mark_skipped)
        return stopped

    def finish_if_exited(self) -> bool:
        """Clear playback state when the managed process has naturally exited."""

        if self.process is None or self.process.poll() is None:
            return False
        self.clear()
        return True

    def current_pid(self) -> int | None:
        """Return the recorded Tonepath mpv PID if it is valid."""

        value = self.store.get_app_state(CURRENT_MPV_PID_KEY)
        return parse_int(value)

    def current_play_id(self) -> int | None:
        """Return the recorded Tonepath play row id if it is valid."""

        value = self.store.get_app_state(CURRENT_PLAY_ID_KEY)
        return parse_int(value)

    def clear(self, mark_skipped: bool = False) -> None:
        """Clear managed playback process, PID state, and active play state."""

        ipc_path = self.active_ipc_path()
        play_id = self.current_play_id()
        if play_id is not None:
            self.store.end_play(play_id, skipped=mark_skipped)
        self.process = None
        self.ipc_path = None
        self.store.delete_app_state(CURRENT_MPV_PID_KEY)
        self.store.delete_app_state(CURRENT_PLAY_ID_KEY)
        self.store.delete_app_state(CURRENT_MPV_IPC_PATH_KEY)
        remove_managed_ipc_socket(ipc_path)


def parse_int(value: str | None) -> int | None:
    """Parse an optional integer state value."""

    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def optional_float(value: object) -> float | None:
    """Convert one mpv property to a float when available."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"mpv returned a non-numeric playback property: {value!r}") from None


def read_optional_property(adapter: MpvAdapter, ipc_path: Path, name: str) -> object:
    """Read an mpv property that may be unavailable while media is loading."""

    try:
        return adapter.send_command(ipc_path, ["get_property", name])
    except MpvCommandError as exc:
        if exc.error == "property unavailable":
            return None
        raise


def new_ipc_path() -> Path:
    """Return a unique workspace-local mpv IPC socket path."""

    run_dir = config.ensure_data_dir() / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"mpv-{uuid4().hex}.sock"


def remove_managed_ipc_socket(path: Path | None) -> None:
    """Remove one Tonepath-generated mpv socket without touching other paths."""

    if path is None or path.parent != config.data_dir() / "run":
        return
    name = path.name
    if not name.startswith("mpv-") or not name.endswith(".sock"):
        return
    token = name[len("mpv-") : -len(".sock")]
    if len(token) != 32 or any(character not in "0123456789abcdef" for character in token):
        return
    path.unlink(missing_ok=True)
