import signal
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.playback import MpvAdapter


class FakeInterruptProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            raise KeyboardInterrupt
        return 0

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class PlaybackTest(unittest.TestCase):
    def test_build_command_for_dry_run(self) -> None:
        command = MpvAdapter().build_command([Path("/tmp/a.mp3")])
        self.assertEqual(command[:4], ["mpv", "--no-terminal", "--force-window=no", "--audio-display=no"])
        self.assertEqual(command[-1], "/tmp/a.mp3")

    def test_foreground_playback_terminates_on_keyboard_interrupt(self) -> None:
        process = FakeInterruptProcess()
        adapter = MpvAdapter()
        with self.assertRaises(KeyboardInterrupt):
            adapter.wait_and_stop_on_interrupt(process)  # type: ignore[arg-type]
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_stop_pid_sends_sigterm(self) -> None:
        with patch("tonepath.playback.os.kill", side_effect=[None, ProcessLookupError]):
            stopped = MpvAdapter().stop_pid(1234)
        self.assertTrue(stopped)

    def test_stop_missing_pid_returns_false(self) -> None:
        with patch("tonepath.playback.os.kill", side_effect=ProcessLookupError):
            stopped = MpvAdapter().stop_pid(1234)
        self.assertFalse(stopped)

    def test_stop_pid_uses_sigterm_first(self) -> None:
        with patch("tonepath.playback.os.kill", side_effect=ProcessLookupError) as kill:
            MpvAdapter().stop_pid(1234)
        kill.assert_called_once_with(1234, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
