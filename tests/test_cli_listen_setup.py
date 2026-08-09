import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath import config
from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.playback_controller import CURRENT_MPV_PID_KEY


class CliListenSetupTest(unittest.TestCase):
    def test_setup_private_writes_local_first_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                result = CliRunner().invoke(app, ["setup", "--preset", "private", "--music-dir", str(music)])

                self.assertEqual(result.exit_code, 0, result.output)
                settings = config.load_config()
                self.assertEqual(settings.experience.mode, "private")
                self.assertEqual(settings.music_dirs, (str(music),))
                self.assertFalse(settings.privacy.send_to_llm)
                self.assertEqual(settings.network_mode, "offline")

    def test_setup_smart_dry_run_does_not_write_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                result = CliRunner().invoke(app, ["setup", "--preset", "smart", "--dry-run"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("[experience]", result.output)
                self.assertIn('mode = "smart"', result.output)
                self.assertFalse((home / "config.toml").exists())

    def test_first_run_setup_uses_three_step_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                result = CliRunner().invoke(
                    app,
                    ["setup"],
                    input=f"{music}\nprivate\ny\nn\n",
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Getting Started", result.output)
                self.assertIn("Step 1 of 3 · Music", result.output)
                self.assertIn("Step 2 of 3 · Experience", result.output)
                self.assertIn("Step 3 of 3 · Review & Start", result.output)
                settings = config.load_config()
                self.assertEqual(settings.music_dirs, (str(music),))
                self.assertEqual(settings.experience.mode, "private")
                self.assertFalse(settings.privacy.send_to_llm)

    def test_unconfirmed_first_run_setup_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                result = CliRunner().invoke(
                    app,
                    ["setup"],
                    input=f"{music}\nprivate\nn\n",
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Setup cancelled; no changes were saved.", result.output)
                self.assertFalse(config.config_path().exists())

    def test_first_run_setup_rejects_missing_music_directory_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            missing = Path(tmp) / "missing"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                result = CliRunner().invoke(app, ["setup"], input=f"{missing}\n")

                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("Music directory does not exist", result.output)
                self.assertFalse(config.config_path().exists())

    def test_first_run_setup_dry_run_prints_review_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                result = CliRunner().invoke(
                    app,
                    ["setup", "--dry-run"],
                    input=f"{music}\nprivate\ny\n",
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Would write config", result.output)
                self.assertFalse(config.config_path().exists())

    def test_smart_setup_requires_separate_text_consent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                result = CliRunner().invoke(
                    app,
                    ["setup"],
                    input=f"{music}\nsmart\nn\ny\nn\n",
                )

                self.assertEqual(result.exit_code, 0, result.output)
                settings = config.load_config()
                self.assertEqual(settings.experience.mode, "smart")
                self.assertFalse(settings.privacy.send_to_llm)

    def test_reconfigure_changes_only_selected_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                current = config.preset_config("private", music_dir=first)
                config.write_config(current)
                result = CliRunner().invoke(
                    app,
                    ["setup"],
                    input="experience\nsmart\nn\ndone\ny\nn\n",
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Current Tonepath setup", result.output)
                settings = config.load_config()
                self.assertEqual(settings.music_dirs, (str(first),))
                self.assertEqual(settings.experience.mode, "smart")
                self.assertFalse(settings.privacy.send_to_llm)

    def test_noninteractive_setup_adds_music_directory_without_clearing_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(config.preset_config("private", music_dir=first))

                result = CliRunner().invoke(
                    app,
                    ["setup", "--preset", "smart", "--music-dir", str(second), "--llm-provider", "qwen"],
                )

                self.assertEqual(result.exit_code, 0, result.output)
                settings = config.load_config()
                self.assertEqual(settings.music_dirs, (str(first), str(second)))
                self.assertEqual(settings.llm.provider, "qwen")
                self.assertFalse(settings.privacy.send_to_llm)

    def test_prepare_and_model_setup_use_separate_confirmations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}), patch(
                "tonepath.cli.model_runtime_status",
                return_value=SimpleNamespace(ready=False, affect_ready=False),
            ), patch("tonepath.cli.run_cli_preparation") as run_prepare:
                result = CliRunner().invoke(
                    app,
                    ["setup"],
                    input=f"{music}\nprivate\ny\ny\nn\n",
                )

                self.assertEqual(result.exit_code, 0, result.output)
                run_prepare.assert_called_once()
                self.assertFalse(run_prepare.call_args.kwargs["setup_models"])

    def test_listen_without_tracks_gives_setup_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                result = CliRunner().invoke(app, ["listen", "focus 30m no vocals", "--dry-run"])

                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn("Tonepath experience: Private", result.output)
                self.assertIn("No prepared library yet.", result.output)
                self.assertIn("prepare", result.output)

    def test_listen_review_files_does_not_generate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(config.preset_config("private", music_dir=music))
                store = TonepathStore()
                path = music / "song.mp3"
                path.write_bytes(b"fake")
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
                good_path = music / "ready.mp3"
                good_path.write_bytes(b"fake")
                ready_id = store.upsert_track(
                    Track(
                        id=None,
                        path=good_path,
                        file_hash="ready",
                        mtime=1.0,
                        title="ready",
                        artist="artist",
                        album=None,
                        genre=None,
                        duration=180.0,
                        format="mp3",
                    )
                )
                store.upsert_features(
                    TrackFeatures(
                        track_id=ready_id,
                        bpm=92.0,
                        loudness=-15.0,
                        energy=0.35,
                        vocalness=0.1,
                        feature_source="test",
                        confidence="high",
                    )
                )
                store.close()
                result = CliRunner().invoke(app, ["listen", "focus 30m no vocals", "--dry-run"])

            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("Readiness: Review files", result.output)
            self.assertIn("Review or replace files", result.output)
            self.assertNotIn("Dry-run mpv command", result.output)

    def test_listen_dry_run_uses_existing_selection_without_storing_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                self.add_ready_track(tmp)
                result = CliRunner().invoke(app, ["listen", "我要写论文，四十五分钟，低刺激，最好不要人声", "--dry-run"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Tonepath experience: Private", result.output)
                self.assertIn("Dry-run mpv command:", result.output)
                store = TonepathStore()
                self.assertIsNone(store.get_app_state(CURRENT_MPV_PID_KEY))
                self.assertIsNone(store.current_session_id())
                store.close()

    def test_listen_smart_falls_back_when_llm_key_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "TONEPATH_LLM_PROVIDER": "deepseek"}, clear=True):
                config.write_config(config.preset_config("smart", send_to_llm=True))
                self.add_ready_track(tmp)
                result = CliRunner().invoke(app, ["listen", "focus 30m no vocals", "--dry-run"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Tonepath experience: Smart", result.output)
                self.assertIn("DEEPSEEK_API_KEY missing; using deterministic parser", result.output)

    def add_ready_track(self, tmp: str) -> None:
        current = config.load_config()
        config.write_config(
            config.TonepathConfig(
                music_dirs=(str(Path(tmp)),),
                data_dir=current.data_dir,
                player=current.player,
                network_mode=current.network_mode,
                privacy=current.privacy,
                models=current.models,
                experience=current.experience,
            )
        )
        store = TonepathStore()
        path = Path(tmp) / "song.mp3"
        path.write_bytes(b"fake")
        track_id = store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash="hash",
                mtime=1.0,
                title="song",
                artist="artist",
                album=None,
                genre="instrumental",
                duration=180.0,
                format="mp3",
            )
        )
        store.upsert_features(
            TrackFeatures(
                track_id=track_id,
                bpm=92.0,
                loudness=-15.0,
                energy=0.35,
                vocalness=0.1,
                feature_source="test",
                confidence="high",
            )
        )
        store.close()


if __name__ == "__main__":
    unittest.main()
