import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath import config
from tonepath.doctor import run_doctor


class ConfigTest(unittest.TestCase):
    def test_init_config_writes_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": tmp}):
                path = config.init_config()
                self.assertEqual(path, Path(tmp) / "config.toml")
                self.assertTrue(path.exists())
                settings = config.load_config()
                self.assertEqual(settings.music_dirs, ("~/Music",))
                self.assertEqual(settings.player, "mpv")
                self.assertEqual(settings.network_mode, "offline")
                self.assertFalse(settings.privacy.send_to_llm)
                self.assertTrue(settings.privacy.store_play_history)
                self.assertEqual(settings.models.mode, "balanced")
                self.assertFalse(settings.models.allow_setup)
                self.assertFalse(settings.models.allow_online)
                self.assertEqual(settings.models.preferred_tagger, "essentia-tf")
                self.assertEqual(settings.models.separator_fallback, "off")

    def test_default_home_is_workspace_local_when_not_overridden(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.app_home(), config.repo_root().parent / ".tonepath")

    def test_add_music_dir_persists_expanded_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                music_dir = Path(tmp) / "音乐"
                music_dir.mkdir()
                config.init_config()
                settings = config.add_music_dir(music_dir)
                self.assertIn(str(music_dir), settings.music_dirs)
                self.assertIn(str(music_dir), config.load_config().music_dirs)

    def test_doctor_reports_missing_music_dirs_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tonepath_home = Path(tmp) / "home"
            missing = Path(tmp) / "missing"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(tonepath_home)}):
                settings = config.TonepathConfig(
                    music_dirs=(str(missing),),
                    data_dir=str(tonepath_home),
                    player="mpv",
                    network_mode="offline",
                    privacy=config.PrivacyConfig(),
                    models=config.ModelConfig(),
                )
                config.write_config(settings)
                report = run_doctor()
                self.assertIn(f"{missing}: missing", report)
                self.assertIn(f"Config path: {tonepath_home / 'config.toml'}", report)
                self.assertIn("Player: mpv", report)
                self.assertIn("Network mode: offline", report)


if __name__ == "__main__":
    unittest.main()
