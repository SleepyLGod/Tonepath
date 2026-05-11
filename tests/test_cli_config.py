import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app


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
            with patch.dict(os.environ, {"TONEPATH_HOME": str(tonepath_home)}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                add_result = runner.invoke(app, ["config", "add-music-dir", str(music_dir)])
                self.assertEqual(add_result.exit_code, 0, add_result.output)
                scan_result = runner.invoke(app, ["scan"])
                self.assertEqual(scan_result.exit_code, 0, scan_result.output)
                self.assertIn("Scanned 1 track(s)", scan_result.output)


if __name__ == "__main__":
    unittest.main()
