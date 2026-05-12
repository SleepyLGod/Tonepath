"""Playback lifecycle management for Tonepath."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.playback import MpvAdapter


CURRENT_MPV_PID_KEY = "current_mpv_pid"
CURRENT_PLAY_ID_KEY = "current_play_id"


class PlaybackController:
    """Own Tonepath-managed playback process state."""

    def __init__(self, store: TonepathStore, adapter: MpvAdapter | None = None) -> None:
        self.store = store
        self.adapter = adapter or MpvAdapter()
        self.process: subprocess.Popen[bytes] | None = None

    def start(
        self,
        paths: list[Path],
        session_id: int | None = None,
        track_id: int | None = None,
    ) -> subprocess.Popen[bytes]:
        """Start playback and store the managed process PID."""

        process = self.adapter.start(paths)
        self.process = process
        self.store.set_app_state(CURRENT_MPV_PID_KEY, str(process.pid))
        if track_id is not None:
            play_id = self.store.start_play(session_id=session_id, track_id=track_id)
            self.store.set_app_state(CURRENT_PLAY_ID_KEY, str(play_id))
        return process

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

        play_id = self.current_play_id()
        if play_id is not None:
            self.store.end_play(play_id, skipped=mark_skipped)
        self.process = None
        self.store.delete_app_state(CURRENT_MPV_PID_KEY)
        self.store.delete_app_state(CURRENT_PLAY_ID_KEY)


def parse_int(value: str | None) -> int | None:
    """Parse an optional integer state value."""

    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
