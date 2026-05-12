import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.db import TonepathStore
from tonepath.models import Track
from tonepath.playback import MpvAdapter
from tonepath.tui import TonepathApp


class FakeProcess:
    pid = 9876

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


class FinishedProcess(FakeProcess):
    def poll(self) -> int | None:
        return 0


class TonepathTuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_tui_launches_session_screen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        self.assertIsNotNone(app.query_one("#timeline"))
                        self.assertIsNotNone(app.query_one("#queue"))
                        self.assertIsNotNone(app.query_one("#why-panel"))
                        self.assertIsNotNone(app.query_one("#event-log"))
                        await pilot.press("w")
                        await pilot.press("s")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 0)

    async def test_tui_play_starts_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 1)

    async def test_tui_quit_stops_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("q")
                self.assertTrue(stop.called)

    async def test_tui_skip_replaces_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start, patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("s")
                        await pilot.press("q")
                self.assertGreaterEqual(start.call_count, 2)
                self.assertTrue(stop.called)

    async def test_tui_stop_key_stops_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("p")
                        await pilot.press("x")
                        await pilot.press("q")
                self.assertTrue(stop.called)

    async def test_tui_natural_finish_records_play_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FinishedProcess()):
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        app.poll_playback_finished()
                        await pilot.press("q")

                store = TonepathStore()
                row = store.conn.execute("SELECT ended_at, skipped FROM plays").fetchone()
                self.assertIsNotNone(row["ended_at"])
                self.assertEqual(row["skipped"], 0)
                store.close()

    def add_track(self, store: TonepathStore, tmp: str, name: str) -> int:
        path = Path(tmp) / name
        path.write_bytes(b"not real audio")
        return store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash=name,
                mtime=1.0,
                title=name,
                artist="artist",
                album=None,
                genre=None,
                duration=None,
                format="mp3",
            )
        )


if __name__ == "__main__":
    unittest.main()
