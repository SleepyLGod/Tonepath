import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.scanner import read_track


class CliConfigTest(unittest.TestCase):
    def test_config_show_reads_initialized_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                runner = CliRunner()
                init_result = runner.invoke(app, ["config", "init"])
                self.assertEqual(init_result.exit_code, 0, init_result.output)
                show_result = runner.invoke(app, ["config", "show"])
                self.assertEqual(show_result.exit_code, 0, show_result.output)
                self.assertIn("music_dirs = [\"~/Music\"]", show_result.output)
                self.assertIn("network_mode = \"offline\"", show_result.output)
                self.assertIn("[privacy]", show_result.output)

    def test_scan_without_argument_uses_configured_music_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tonepath_home = Path(tmp) / "home"
            music_dir = Path(tmp) / "音乐"
            music_dir.mkdir()
            (music_dir / "测试歌曲.mp3").write_bytes(b"not real audio but acceptable for fallback")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(tonepath_home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                add_result = runner.invoke(app, ["config", "add-music-dir", str(music_dir)])
                self.assertEqual(add_result.exit_code, 0, add_result.output)
                scan_result = runner.invoke(app, ["scan"])
                self.assertEqual(scan_result.exit_code, 0, scan_result.output)
                self.assertIn("Scanned 1 track(s)", scan_result.output)

    def test_scan_prunes_missing_tracks_only_under_scanned_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tonepath_home = Path(tmp) / "home"
            music_dir = Path(tmp) / "music"
            other_dir = Path(tmp) / "other"
            music_dir.mkdir()
            other_dir.mkdir()
            stale = music_dir / "stale.mp3"
            current = music_dir / "current.mp3"
            outside = other_dir / "outside.mp3"
            stale.write_bytes(b"old")
            current.write_bytes(b"current")
            outside.write_bytes(b"outside")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(tonepath_home)}):
                store = TonepathStore()
                stale_id = store.upsert_track(read_track(stale))
                outside_id = store.upsert_track(read_track(outside))
                store.close()
                stale.unlink()

                scan_result = CliRunner().invoke(app, ["scan", str(music_dir)])
                self.assertEqual(scan_result.exit_code, 0, scan_result.output)
                self.assertIn("Pruned 1 missing track(s).", scan_result.output)

                store = TonepathStore()
                self.assertIsNone(store.get_track(stale_id))
                self.assertIsNotNone(store.get_track(outside_id))
                self.assertEqual(len(store.list_tracks()), 2)
                store.close()


if __name__ == "__main__":
    unittest.main()
