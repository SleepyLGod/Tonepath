import tempfile
import unittest
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.models import Track
from tonepath.playback_controller import CURRENT_MPV_PID_KEY, CURRENT_PLAY_ID_KEY, PlaybackController


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminated = False


class FakeAdapter:
    def __init__(self) -> None:
        self.started: list[list[Path]] = []
        self.stopped_processes: list[FakeProcess] = []
        self.stopped_pids: list[int] = []
        self.next_pid = 100

    def start(self, paths: list[Path]) -> FakeProcess:
        self.started.append(paths)
        self.next_pid += 1
        return FakeProcess(self.next_pid)

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
