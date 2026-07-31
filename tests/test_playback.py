import json
import signal
import socket
import subprocess
import tempfile
import threading
import unittest
from inspect import signature
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


class FakeExitedProcess(FakeInterruptProcess):
    def poll(self) -> int | None:
        return 1


class PlaybackTest(unittest.TestCase):
    def test_build_command_accepts_ipc_path_and_volume(self) -> None:
        parameters = signature(MpvAdapter.build_command).parameters
        self.assertIn("ipc_path", parameters)
        self.assertIn("volume", parameters)
        command = MpvAdapter().build_command(
            [Path("/tmp/a.mp3")],
            ipc_path=Path("/tmp/tonepath.sock"),
            volume=65.0,
        )
        self.assertIn("--input-ipc-server=/tmp/tonepath.sock", command)
        self.assertIn("--volume=65", command)

    def test_build_command_for_dry_run(self) -> None:
        command = MpvAdapter().build_command([Path("/tmp/a.mp3")])
        self.assertEqual(command[:4], ["mpv", "--no-terminal", "--force-window=no", "--audio-display=no"])
        self.assertEqual(command[-1], "/tmp/a.mp3")

    def test_start_suppresses_child_process_output(self) -> None:
        with patch.object(MpvAdapter, "available", return_value=True), patch("tonepath.playback.subprocess.Popen") as popen:
            MpvAdapter().start([Path("/tmp/a.mp3")])
        popen.assert_called_once()
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)

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

    def test_send_command_round_trips_json_over_unix_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "mpv.sock"
            requests: list[dict[str, object]] = []
            ready = self.start_ipc_server(
                socket_path,
                b'{"error":"success","data":42.5,"request_id":1}\n',
                requests,
            )
            ready.wait(timeout=1.0)

            result = MpvAdapter().send_command(socket_path, ["get_property", "time-pos"])

            self.assertEqual(result, 42.5)
            self.assertEqual(
                requests,
                [{"command": ["get_property", "time-pos"], "request_id": 1}],
            )

    def test_send_command_ignores_events_and_matches_its_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "mpv.sock"
            ready = self.start_ipc_server(
                socket_path,
                b'{"event":"property-change"}\n'
                b'{"error":"success","data":42.5,"request_id":1}\n',
                [],
            )
            ready.wait(timeout=1.0)

            result = MpvAdapter().send_command(socket_path, ["get_property", "time-pos"])

            self.assertEqual(result, 42.5)

    def test_send_command_rejects_malformed_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "mpv.sock"
            ready = self.start_ipc_server(socket_path, b"not-json\n", [])
            ready.wait(timeout=1.0)

            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                MpvAdapter().send_command(socket_path, ["get_property", "pause"])

    def test_send_command_reports_mpv_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "mpv.sock"
            ready = self.start_ipc_server(
                socket_path,
                b'{"error":"property unavailable","request_id":1}\n',
                [],
            )
            ready.wait(timeout=1.0)

            with self.assertRaisesRegex(RuntimeError, "property unavailable"):
                MpvAdapter().send_command(socket_path, ["get_property", "pause"])

    def test_wait_for_ipc_retries_until_command_succeeds(self) -> None:
        adapter = MpvAdapter()
        with patch.object(
            adapter,
            "send_command",
            side_effect=[RuntimeError("not ready"), False],
        ) as send_command, patch("tonepath.playback.time.sleep"):
            adapter.wait_for_ipc(Path("/tmp/mpv.sock"), FakeInterruptProcess(), timeout=1.0)  # type: ignore[arg-type]
        self.assertEqual(send_command.call_count, 2)

    def test_wait_for_ipc_reports_timeout(self) -> None:
        adapter = MpvAdapter()
        with patch.object(adapter, "send_command", side_effect=RuntimeError("not ready")), patch(
            "tonepath.playback.time.monotonic",
            side_effect=[0.0, 0.01, 0.2],
        ), patch("tonepath.playback.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                adapter.wait_for_ipc(Path("/tmp/mpv.sock"), FakeInterruptProcess(), timeout=0.1)  # type: ignore[arg-type]

    def test_wait_for_ipc_reports_early_process_exit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exited"):
            MpvAdapter().wait_for_ipc(Path("/tmp/mpv.sock"), FakeExitedProcess(), timeout=0.1)  # type: ignore[arg-type]

    def start_ipc_server(
        self,
        socket_path: Path,
        response: bytes,
        requests: list[dict[str, object]],
    ) -> threading.Event:
        ready = threading.Event()

        def serve() -> None:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(socket_path))
                server.listen(1)
                ready.set()
                connection, _ = server.accept()
                with connection:
                    payload = b""
                    while not payload.endswith(b"\n"):
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        payload += chunk
                    if payload:
                        requests.append(json.loads(payload))
                    connection.sendall(response)

        threading.Thread(target=serve, daemon=True).start()
        return ready


if __name__ == "__main__":
    unittest.main()
