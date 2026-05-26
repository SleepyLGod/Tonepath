import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track
from tonepath.playback import MpvAdapter
from tonepath.playback_controller import CURRENT_MPV_PID_KEY


class FakeProcess:
    pid = 4321


class CliPlaybackTest(unittest.TestCase):
    def test_stop_without_active_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                result = CliRunner().invoke(app, ["stop"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("No active Tonepath playback.", result.output)

    def test_start_dry_run_does_not_store_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tonepath_home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(tonepath_home)}):
                self.add_track(tmp)
                result = CliRunner().invoke(app, ["start", "from irritated to focus in 30 minutes", "--dry-run"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Dry-run mpv command:", result.output)
                store = TonepathStore()
                self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
                self.assertIsNone(store.current_session_id())
                store.close()

    def test_start_background_stores_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tonepath_home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(tonepath_home)}):
                self.add_track(tmp)
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()):
                    result = CliRunner().invoke(app, ["start", "from irritated to focus in 30 minutes", "--background"])
                self.assertEqual(result.exit_code, 0, result.output)
                store = TonepathStore()
                self.assertEqual(store.get_app_state(CURRENT_MPV_PID_KEY), "4321")
                store.close()

    def test_stop_clears_stored_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.set_app_state(CURRENT_MPV_PID_KEY, "4321")
                store.close()
                with patch.object(MpvAdapter, "stop_pid", return_value=True):
                    result = CliRunner().invoke(app, ["stop"])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Stopped Tonepath playback PID 4321.", result.output)
                store = TonepathStore()
                self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
                store.close()

    def add_track(self, tmp: str) -> None:
        store = TonepathStore()
        path = Path(tmp) / "song.mp3"
        path.write_bytes(b"not real audio")
        store.upsert_track(
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
        store.close()


if __name__ == "__main__":
    unittest.main()
