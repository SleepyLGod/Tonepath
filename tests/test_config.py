import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tonepath import config
from tonepath.doctor import run_doctor
from tonepath.tui_theme import PALETTE_BY_KEY, PALETTES, normalize_theme


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
                self.assertEqual(settings.experience.mode, "private")
                self.assertEqual(settings.ui.theme, "warmline")

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
                    experience=config.ExperienceConfig(),
                )
                config.write_config(settings)
                report = run_doctor()
                self.assertIn(f"{missing}: missing", report)
                self.assertIn(f"Config path: {tonepath_home / 'config.toml'}", report)
                self.assertIn("Player: mpv", report)
                self.assertIn("Network mode: offline", report)

    def test_preset_config_private_is_local_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                music_dir = Path(tmp) / "music"
                settings = config.preset_config("private", music_dir=music_dir)

                self.assertEqual(settings.experience.mode, "private")
                self.assertEqual(settings.music_dirs, (str(music_dir),))
                self.assertEqual(settings.network_mode, "offline")
                self.assertFalse(settings.privacy.send_to_llm)
                self.assertEqual(settings.models.mode, "balanced")
                self.assertFalse(settings.models.allow_online)

    def test_preset_config_music_dir_replaces_existing_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(replace(config.default_config(), music_dirs=(str(first), str(second))))
                settings = config.preset_config("private", music_dir=first)

                self.assertEqual(settings.music_dirs, (str(first),))

    def test_preset_config_smart_is_opt_in_intelligent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                settings = config.preset_config("smart")

                self.assertEqual(settings.experience.mode, "smart")
                self.assertEqual(settings.network_mode, "online-opt-in")
                self.assertTrue(settings.privacy.send_to_llm)
                self.assertEqual(settings.models.mode, "full")
                self.assertTrue(settings.models.allow_online)
                self.assertFalse(settings.models.allow_setup)

    def test_ui_theme_loads_and_invalid_theme_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                settings = config.write_config(replace(config.default_config(), ui=config.UiConfig(theme="midnight")))
                self.assertTrue(settings.exists())
                self.assertEqual(config.load_config().ui.theme, "midnight")

                (Path(tmp) / "home" / "config.toml").write_text(
                    "\n".join(
                        [
                            'music_dirs = ["~/Music"]',
                            'data_dir = "x"',
                            'player = "mpv"',
                            'network_mode = "offline"',
                            "",
                            "[ui]",
                            'theme = "unknown"',
                        ]
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(config.load_config().ui.theme, "warmline")

    def test_builtin_tui_theme_pack_is_stable(self) -> None:
        keys = [palette.key for palette in PALETTES]

        self.assertEqual(len(PALETTES), 9)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(keys[:3], ["warmline", "midnight", "high-contrast"])
        self.assertEqual(normalize_theme("solarized-light"), "solarized-light")
        self.assertEqual(normalize_theme("catppuccin-mocha"), "catppuccin-mocha")
        self.assertEqual(normalize_theme("unknown"), "warmline")
        self.assertFalse(PALETTE_BY_KEY["solarized-light"].dark)
        self.assertFalse(PALETTE_BY_KEY["catppuccin-latte"].dark)
        self.assertTrue(PALETTE_BY_KEY["dracula"].dark)
        self.assertTrue(PALETTE_BY_KEY["jukebox"].dark)


if __name__ == "__main__":
    unittest.main()
