import os
import tempfile
import unittest
from pathlib import Path
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
                config.write_config(config.preset_config("smart"))
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
