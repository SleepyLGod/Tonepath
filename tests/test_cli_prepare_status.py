import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath import config
from tonepath.analysis import AnalysisProgress
from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.scanner import read_track


class CliPrepareStatusTest(unittest.TestCase):
    def test_prepare_runs_scan_mir_and_tags_when_runtime_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            (music / "song.mp3").write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                self.assertEqual(runner.invoke(app, ["config", "add-music-dir", str(music)]).exit_code, 0)
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=True)), patch(
                    "tonepath.cli.analyze_library",
                    side_effect=[(1, 0), (1, 0)],
                ) as analyze:
                    result = runner.invoke(app, ["prepare", "--limit", "5"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Prepare: scan", result.output)
            self.assertIn("Prepare: MIR analyzed 1 track(s); skipped 0 track(s).", result.output)
            self.assertIn("Prepare: tags analyzed 1 track(s); skipped 0 track(s).", result.output)
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(analyze.call_args_list[0].kwargs["features"], "mir")
            self.assertEqual(analyze.call_args_list[0].kwargs["changed_only"], True)
            self.assertEqual(analyze.call_args_list[0].kwargs["limit"], 5)
            self.assertEqual(analyze.call_args_list[1].kwargs["features"], "tags")
            self.assertEqual(analyze.call_args_list[1].kwargs["method"], "essentia-tf")

    def test_prepare_skips_tags_when_runtime_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            (music / "song.mp3").write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                self.assertEqual(runner.invoke(app, ["config", "add-music-dir", str(music)]).exit_code, 0)
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=False)), patch(
                    "tonepath.cli.analyze_library",
                    return_value=(1, 0),
                ) as analyze:
                    result = runner.invoke(app, ["prepare"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("tags skipped", result.output)
            self.assertIn("models setup essentia-tf", result.output)
            self.assertEqual(analyze.call_count, 1)

    def test_prepare_full_without_runtime_prints_setup_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            (music / "song.mp3").write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                self.assertEqual(runner.invoke(app, ["config", "add-music-dir", str(music)]).exit_code, 0)
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=False)), patch(
                    "tonepath.cli.analyze_library",
                    return_value=(1, 0),
                ) as analyze:
                    result = runner.invoke(app, ["prepare", "--full"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("full tagging requires Essentia-TF", result.output)
            self.assertIn("prepare --full", result.output)
            self.assertEqual(analyze.call_count, 1)

    def test_prepare_setup_models_creates_runtime_before_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            (music / "song.mp3").write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                self.assertEqual(runner.invoke(app, ["config", "add-music-dir", str(music)]).exit_code, 0)
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=False)), patch(
                    "tonepath.cli.setup_essentia_tf_runtime",
                    return_value=SimpleNamespace(ready=True),
                ) as setup, patch(
                    "tonepath.cli.analyze_library",
                    side_effect=[(1, 0), (1, 0)],
                ) as analyze:
                    result = runner.invoke(app, ["prepare", "--full", "--setup-models"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("setting up workspace-local Essentia-TF runtime", result.output)
            setup.assert_called_once()
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(analyze.call_args_list[1].kwargs["features"], "tags")

    def test_prepare_fast_skips_tags_even_when_runtime_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            (music / "song.mp3").write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                self.assertEqual(runner.invoke(app, ["config", "add-music-dir", str(music)]).exit_code, 0)
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=True)), patch(
                    "tonepath.cli.analyze_library",
                    return_value=(1, 0),
                ) as analyze:
                    result = runner.invoke(app, ["prepare", "--fast"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("skipped TensorFlow tags (--fast)", result.output)
            self.assertEqual(analyze.call_count, 1)

    def test_prepare_prints_failed_analysis_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            bad_track = track_for(music / "bad.mp3", title="Bad", artist=None)

            def fake_analyze(*args: object, **kwargs: object) -> tuple[int, int]:
                progress = kwargs.get("progress")
                if callable(progress):
                    progress(AnalysisProgress(1, 1, bad_track, error="Essentia MIR analysis failed for bad.mp3."))
                return (0, 1)

            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "HOME": str(Path(tmp) / "user-home")}):
                runner = CliRunner()
                self.assertEqual(runner.invoke(app, ["config", "init"]).exit_code, 0)
                self.assertEqual(runner.invoke(app, ["config", "add-music-dir", str(music)]).exit_code, 0)
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=True)), patch(
                    "tonepath.cli.analyze_library",
                    side_effect=fake_analyze,
                ), patch("tonepath.cli.audio_probe_diagnostic", return_value="invalid audio: no MPEG frames found"):
                    result = runner.invoke(app, ["prepare"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("some files could not be analyzed", result.output)
            self.assertIn("Bad - unknown", result.output)
            self.assertIn("invalid audio: no MPEG frames found", result.output)

    def test_status_prints_counts_without_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "song.mp3"
            music.write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret-value"}):
                store = TonepathStore()
                track_id = store.upsert_track(read_track(music))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        bpm=100.0,
                        loudness=-14.0,
                        energy=0.5,
                        vocalness=0.2,
                        feature_source="test",
                        confidence="high",
                    )
                )
                store.close()
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=True)):
                    result = CliRunner().invoke(app, ["status"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Tracks", result.output)
            self.assertIn("Vocalness coverage", result.output)
            self.assertIn("Tag coverage", result.output)
            self.assertIn("Dirty metadata", result.output)
            self.assertIn("Duplicate candidates", result.output)
            self.assertIn("Tracks outside music dirs", result.output)
            self.assertIn("Model mode", result.output)
            self.assertIn("Next action", result.output)
            self.assertIn("ready", result.output)
            self.assertNotIn("secret-value", result.output)

    def test_status_lists_missing_analysis_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(
                    config.TonepathConfig(
                        music_dirs=(str(music),),
                        data_dir=str(home),
                        player="mpv",
                        network_mode="offline",
                        privacy=config.PrivacyConfig(),
                        models=config.ModelConfig(),
                    )
                )
                store = TonepathStore()
                good_id = store.upsert_track(track_for(music / "good.mp3", title="Good", artist="Artist"))
                store.upsert_features(
                    TrackFeatures(
                        good_id,
                        bpm=100.0,
                        loudness=-14.0,
                        energy=0.5,
                        vocalness=0.2,
                        feature_source="test",
                        confidence="high",
                    )
                )
                store.upsert_track(track_for(music / "bad.mp3", title="Bad", artist=None))
                store.close()
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=True)):
                    result = CliRunner().invoke(app, ["status"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Missing analysis files", result.output)
            self.assertIn("Bad - unknown", result.output)
            self.assertIn("Review or replace files with missing analysis", result.output)

    def test_status_reports_tracks_outside_configured_dirs_and_dirty_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            configured = Path(tmp) / "configured"
            configured.mkdir()
            outside = Path(tmp) / "songs"
            outside.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(
                    config.TonepathConfig(
                        music_dirs=(str(configured),),
                        data_dir=str(home),
                        player="mpv",
                        network_mode="offline",
                        privacy=config.PrivacyConfig(),
                        models=config.ModelConfig(),
                    )
                )
                store = TonepathStore()
                first_id = store.upsert_track(track_for(outside / "one.mp3", title="Song(null)", artist="unknown"))
                second_id = store.upsert_track(track_for(outside / "two.mp3", title="Song", artist="unknown"))
                for track_id in (first_id, second_id):
                    store.upsert_features(
                        TrackFeatures(
                            track_id=track_id,
                            bpm=100.0,
                            loudness=-14.0,
                            energy=0.5,
                            vocalness=0.2,
                            feature_source="test",
                            confidence="high",
                        )
                    )
                store.close()
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=True)):
                    result = CliRunner().invoke(app, ["status"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Dirty metadata", result.output)
            self.assertIn("2", result.output)
            self.assertIn("Tracks outside music dirs", result.output)
            self.assertIn("config add-music-dir", result.output)

    def test_status_uses_model_policy_for_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "song.mp3"
            music.write_bytes(b"fake audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(
                    config.TonepathConfig(
                        music_dirs=(str(Path(tmp)),),
                        data_dir=str(home),
                        player="mpv",
                        network_mode="offline",
                        privacy=config.PrivacyConfig(),
                        models=config.ModelConfig(mode="full"),
                    )
                )
                store = TonepathStore()
                track_id = store.upsert_track(read_track(music))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        bpm=100.0,
                        loudness=-14.0,
                        energy=0.5,
                        vocalness=None,
                        feature_source="test",
                        confidence="medium",
                    )
                )
                store.close()
                with patch("tonepath.cli.model_runtime_status", return_value=SimpleNamespace(ready=False)):
                    result = CliRunner().invoke(app, ["status"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("full", result.output)
            self.assertIn("models setup essentia-tf", result.output)

def track_for(path: Path, title: str | None, artist: str | None) -> Track:
    """Create one status-test track payload."""

    path.write_bytes(b"fake audio")
    return Track(
        id=None,
        path=path,
        file_hash=path.name,
        mtime=1.0,
        title=title,
        artist=artist,
        album=None,
        genre=None,
        duration=180.0,
        format="mp3",
    )


if __name__ == "__main__":
    unittest.main()
