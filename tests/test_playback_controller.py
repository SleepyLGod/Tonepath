import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.db import TonepathStore
from tonepath.models import Track
from tonepath.playback import MpvCommandError
from tonepath.playback_controller import CURRENT_MPV_PID_KEY, CURRENT_PLAY_ID_KEY, PlaybackController


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminated = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


class FakeAdapter:
    def __init__(self) -> None:
        self.started: list[tuple[list[Path], Path | None, float | None]] = []
        self.stopped_processes: list[FakeProcess] = []
        self.stopped_pids: list[int] = []
        self.commands: list[list[object]] = []
        self.next_pid = 100
        self.wait_error: RuntimeError | None = None
        self.properties: dict[str, object] = {
            "pause": False,
            "time-pos": 12.5,
            "duration": 180.0,
            "volume": 100.0,
        }

    def start(
        self,
        paths: list[Path],
        ipc_path: Path | None = None,
        volume: float | None = None,
    ) -> FakeProcess:
        self.started.append((paths, ipc_path, volume))
        self.next_pid += 1
        return FakeProcess(self.next_pid)

    def wait_for_ipc(self, ipc_path: Path, process: FakeProcess) -> None:
        if self.wait_error is not None:
            raise self.wait_error

    def send_command(self, ipc_path: Path, command: list[object]) -> object:
        self.commands.append(command)
        if command[0] == "get_property":
            return self.properties[str(command[1])]
        if command[0] == "set_property":
            self.properties[str(command[1])] = command[2]
            return None
        if command[0] == "seek":
            self.properties["time-pos"] = float(self.properties["time-pos"]) + float(command[1])
            return None
        raise AssertionError(f"Unexpected command: {command}")

    def stop_process(self, process: FakeProcess) -> None:
        process.terminated = True
        self.stopped_processes.append(process)

    def stop_pid(self, pid: int) -> bool:
        self.stopped_pids.append(pid)
        return True

    def wait_and_stop_on_interrupt(self, process: FakeProcess) -> int:
        return 0


class PlaybackControllerTest(unittest.TestCase):
    def test_start_stores_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            process = controller.start([Path("/tmp/a.mp3")])
            self.assertEqual(store.get_app_state(CURRENT_MPV_PID_KEY), str(process.pid))
            self.assertIsNotNone(store.get_app_state("current_mpv_ipc_path"))
            self.assertIsNotNone(adapter.started[0][1])
            self.assertEqual(adapter.started[0][2], 100.0)
            store.close()

    def test_start_failure_stops_uncontrollable_process_and_clears_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            adapter.wait_error = RuntimeError("socket unavailable")
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]

            with self.assertRaisesRegex(RuntimeError, "socket unavailable"):
                controller.start([Path("/tmp/a.mp3")])

            self.assertEqual(len(adapter.stopped_processes), 1)
            self.assertIsNone(controller.process)
            self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
            self.assertIsNone(store.get_app_state("current_mpv_ipc_path"))
            store.close()

    def test_start_failure_removes_generated_ipc_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            ipc_path = self.make_ipc_file(home, "mpv-11111111111111111111111111111111.sock")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True), patch(
                "tonepath.playback_controller.new_ipc_path", return_value=ipc_path
            ):
                store = TonepathStore(Path(tmp) / "tonepath.db")
                adapter = FakeAdapter()
                adapter.wait_error = RuntimeError("socket unavailable")
                controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]

                with self.assertRaisesRegex(RuntimeError, "socket unavailable"):
                    controller.start([Path("/tmp/a.mp3")])

                self.assertFalse(ipc_path.exists())
                store.close()

    def test_state_reads_live_mpv_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            controller.start([Path("/tmp/a.mp3")])

            state = controller.state()

            self.assertTrue(state.playing)
            self.assertFalse(state.paused)
            self.assertEqual(state.position_sec, 12.5)
            self.assertEqual(state.duration_sec, 180.0)
            self.assertEqual(state.volume, 100.0)
            store.close()

    def test_state_treats_temporarily_unavailable_numeric_properties_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            controller.start([Path("/tmp/a.mp3")])
            original_send_command = adapter.send_command

            def send_command(ipc_path: Path, command: list[object]) -> object:
                if command == ["get_property", "time-pos"]:
                    raise MpvCommandError("property unavailable")
                return original_send_command(ipc_path, command)

            adapter.send_command = send_command  # type: ignore[method-assign]

            state = controller.state()

            self.assertTrue(state.playing)
            self.assertIsNone(state.position_sec)
            self.assertEqual(state.duration_sec, 180.0)
            store.close()

    def test_pause_resume_seek_and_volume_do_not_create_new_play_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp)
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            controller.start([Path(tmp) / "song.mp3"], track_id=track_id)

            controller.pause()
            controller.resume()
            controller.seek_relative(10.0)
            volume = controller.adjust_volume(-35.0)

            play_count = store.conn.execute("SELECT COUNT(*) FROM plays").fetchone()[0]
            feedback_count = store.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            self.assertEqual(play_count, 1)
            self.assertEqual(feedback_count, 0)
            self.assertEqual(volume, 65.0)
            self.assertIn(["set_property", "pause", True], adapter.commands)
            self.assertIn(["set_property", "pause", False], adapter.commands)
            self.assertIn(["seek", 10.0, "relative+exact"], adapter.commands)
            self.assertIn(["set_property", "volume", 65.0], adapter.commands)
            store.close()

    def test_replace_preserves_controller_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            controller.start([Path("/tmp/a.mp3")])
            controller.adjust_volume(-35.0)

            controller.replace([Path("/tmp/b.mp3")])

            self.assertEqual(adapter.started[-1][2], 65.0)
            store.close()

    def test_stop_current_clears_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            process = controller.start([Path("/tmp/a.mp3")])
            stopped = controller.stop_current()
            self.assertTrue(stopped)
            self.assertTrue(process.terminated)
            self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
            store.close()

    def test_replace_stops_old_process_and_starts_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            old = controller.start([Path("/tmp/a.mp3")])
            new = controller.replace([Path("/tmp/b.mp3")])
            self.assertTrue(old.terminated)
            self.assertNotEqual(old.pid, new.pid)
            self.assertEqual(store.get_app_state(CURRENT_MPV_PID_KEY), str(new.pid))
            store.close()

    def test_replace_removes_old_socket_and_keeps_new_socket_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            old_ipc = self.make_ipc_file(home, "mpv-22222222222222222222222222222222.sock")
            new_ipc = self.make_ipc_file(home, "mpv-33333333333333333333333333333333.sock")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True), patch(
                "tonepath.playback_controller.new_ipc_path", side_effect=[old_ipc, new_ipc]
            ):
                store = TonepathStore(Path(tmp) / "tonepath.db")
                controller = PlaybackController(store, adapter=FakeAdapter())  # type: ignore[arg-type]
                controller.start([Path("/tmp/a.mp3")])

                controller.replace([Path("/tmp/b.mp3")])

                self.assertFalse(old_ipc.exists())
                self.assertTrue(new_ipc.exists())
                controller.stop_current()
                self.assertFalse(new_ipc.exists())
                store.close()

    def test_stop_recorded_clears_stale_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            adapter = FakeAdapter()
            store.set_app_state(CURRENT_MPV_PID_KEY, "321")
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            stopped = controller.stop_recorded()
            self.assertTrue(stopped)
            self.assertEqual(adapter.stopped_pids, [321])
            self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
            store.close()

    def test_start_and_stop_records_play_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp)
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            controller.start([Path(tmp) / "song.mp3"], session_id=None, track_id=track_id)
            play_id = store.get_app_state(CURRENT_PLAY_ID_KEY)
            self.assertIsNotNone(play_id)
            controller.stop_current(mark_skipped=True)
            row = store.conn.execute("SELECT ended_at, skipped FROM plays WHERE id = ?", (play_id,)).fetchone()
            self.assertIsNotNone(row["ended_at"])
            self.assertEqual(row["skipped"], 1)
            self.assertIsNone(store.get_app_state(CURRENT_PLAY_ID_KEY))
            store.close()

    def test_finish_if_exited_clears_play_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp)
            adapter = FakeAdapter()
            controller = PlaybackController(store, adapter=adapter)  # type: ignore[arg-type]
            process = controller.start([Path(tmp) / "song.mp3"], session_id=None, track_id=track_id)
            process.returncode = 0
            finished = controller.finish_if_exited()
            play_id = store.get_app_state(CURRENT_PLAY_ID_KEY)
            row = store.conn.execute("SELECT ended_at, skipped FROM plays").fetchone()
            self.assertTrue(finished)
            self.assertIsNone(play_id)
            self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
            self.assertIsNotNone(row["ended_at"])
            self.assertEqual(row["skipped"], 0)
            store.close()

    def test_natural_finish_removes_generated_ipc_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            ipc_path = self.make_ipc_file(home, "mpv-44444444444444444444444444444444.sock")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True), patch(
                "tonepath.playback_controller.new_ipc_path", return_value=ipc_path
            ):
                store = TonepathStore(Path(tmp) / "tonepath.db")
                controller = PlaybackController(store, adapter=FakeAdapter())  # type: ignore[arg-type]
                process = controller.start([Path("/tmp/a.mp3")])
                process.returncode = 0

                self.assertTrue(controller.finish_if_exited())
                self.assertFalse(ipc_path.exists())
                store.close()

    def test_clear_does_not_remove_non_tonepath_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            outside = Path(tmp) / "do-not-remove.sock"
            outside.write_text("external", encoding="utf-8")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = TonepathStore(Path(tmp) / "tonepath.db")
                store.set_app_state("current_mpv_ipc_path", str(outside))
                controller = PlaybackController(store, adapter=FakeAdapter())  # type: ignore[arg-type]

                controller.clear()

                self.assertTrue(outside.exists())
                store.close()

    def make_ipc_file(self, home: Path, name: str) -> Path:
        run_dir = home / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / name
        path.write_text("socket", encoding="utf-8")
        return path

    def add_track(self, store: TonepathStore, tmp: str) -> int:
        path = Path(tmp) / "song.mp3"
        path.write_bytes(b"not real audio")
        return store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash="hash",
                mtime=1.0,
                title="song",
                artist="artist",
                album=None,
                genre=None,
                duration=None,
                format="mp3",
            )
        )


if __name__ == "__main__":
    unittest.main()
