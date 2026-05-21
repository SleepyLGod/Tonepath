import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(output: str) -> str:
    return ANSI_RE.sub("", output)


class CliAnalyzeTest(unittest.TestCase):
    def test_analyze_basic_stores_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
                track_id = store.upsert_track(
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

                result = CliRunner().invoke(app, ["analyze", "--features", "basic"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Analyzed 1 track(s); skipped 0 track(s).", result.output)
                store = TonepathStore()
                self.assertIsNotNone(store.get_features(track_id))
                store.close()

    def test_analyze_vocalness_command_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
                track_id = store.upsert_track(
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

                with patch("tonepath.analysis.decode_pcm_with_ffmpeg", return_value=None):
                    result = CliRunner().invoke(app, ["analyze", "--features", "vocalness"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Analyzed 1 track(s); skipped 0 track(s).", result.output)
                store = TonepathStore()
                features = store.get_features(track_id)
                self.assertIsNotNone(features)
                self.assertIsNone(features.vocalness)
                store.close()

    def test_analyze_vocalness_demucs_missing_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
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

                with patch("tonepath.analysis.shutil.which", return_value=None):
                    result = CliRunner().invoke(app, ["analyze", "--features", "vocalness", "--method", "demucs-cli"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("demucs-cli vocalness requires", result.output)

    def test_analyze_vocalness_audio_separator_missing_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
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

                with patch("tonepath.analysis.shutil.which", return_value=None):
                    result = CliRunner().invoke(app, ["analyze", "--features", "vocalness", "--method", "audio-separator"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("uv sync --extra models", plain_output(result.output))

    def test_analyze_rejects_model_method_for_basic_features(self) -> None:
        result = CliRunner().invoke(app, ["analyze", "--features", "basic", "--method", "audio-separator"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--method is only supported", plain_output(result.output))
        self.assertIn("tags", result.output)

    def test_analyze_mir_command_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
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

                with patch("tonepath.analysis.import_essentia_standard", return_value=object()), patch(
                    "tonepath.analysis.extract_mir_with_essentia",
                    return_value={"bpm": 100.0, "loudness": -18.0, "key": "C", "scale": "major"},
                ):
                    result = CliRunner().invoke(app, ["analyze", "--features", "mir", "--method", "essentia"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("energy=", result.output)
                self.assertIn("bpm=100.0", result.output)

    def test_analyze_tags_missing_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
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

                with patch("tonepath.analysis.ensure_essentia_tagging_available", side_effect=RuntimeError("TensorFlow model support")):
                    result = CliRunner().invoke(app, ["analyze", "--features", "tags", "--method", "essentia"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("TensorFlow model support", result.output)

    def test_analyze_tags_essentia_tf_command_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
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

                with patch("tonepath.analysis.ensure_essentia_tf_runtime", return_value=None), patch(
                    "tonepath.analysis.run_essentia_tf_tags",
                    return_value={"vocalness": 0.2, "tags": [["instrumental", 0.9]]},
                ):
                    result = CliRunner().invoke(app, ["analyze", "--features", "tags", "--method", "essentia-tf"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("vocalness=0.20", result.output)

    def test_analyze_limit_prints_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                for index in range(2):
                    path = Path(tmp) / f"song-{index}.mp3"
                    path.write_bytes(b"not decoded as audio")
                    store.upsert_track(
                        Track(
                            id=None,
                            path=path,
                            file_hash=f"hash-{index}",
                            mtime=1.0,
                            title=f"song-{index}",
                            artist="artist",
                            album=None,
                            genre=None,
                            duration=None,
                            format="mp3",
                        )
                    )
                store.close()

                with patch("tonepath.analysis.decode_pcm_with_ffmpeg", return_value=None):
                    result = CliRunner().invoke(app, ["analyze", "--features", "vocalness", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("[1/1] analyzing: song-0 - artist", result.output)
                self.assertIn("Analyzed 1 track(s); skipped 0 track(s).", result.output)

    def test_analyze_keyboard_interrupt_reports_resume_hint(self) -> None:
        with patch("tonepath.cli.analyze_library", side_effect=KeyboardInterrupt()):
            result = CliRunner().invoke(app, ["analyze", "--features", "vocalness"])

        self.assertEqual(result.exit_code, 130)
        self.assertIn("rerun with --only-missing", result.output)
        self.assertIn("resume", result.output)


if __name__ == "__main__":
    unittest.main()
