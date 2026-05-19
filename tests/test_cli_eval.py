import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures


class CliEvalTest(unittest.TestCase):
    def test_eval_selection_json_outputs_stable_feature_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "quiet.mp3", title="quiet", genre="ambient"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        energy=0.24,
                        loudness=-18.2,
                        bpm=84.0,
                        vocalness=0.18,
                        feature_source="model-audio-separator",
                        confidence="high",
                    )
                )
                store.close()

                result = CliRunner().invoke(
                    app,
                    ["eval", "selection", "我现在很烦，想半小时后进入写代码状态，不要人声", "--json"],
                )

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload[0]["phase"], "decompress")
                self.assertEqual(payload[0]["track"]["title"], "quiet")
                self.assertEqual(payload[0]["features"]["source"], "model-audio-separator")
                self.assertEqual(payload[0]["features"]["confidence"], "high")
                self.assertEqual(payload[0]["features"]["vocalness"], 0.18)
                self.assertIn("vocalness feature supports no-vocals constraint", payload[0]["reasons"])

    def test_eval_selection_does_not_write_profile_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                before = store.profile_summary()
                store.close()

                result = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                store = TonepathStore()
                after = store.profile_summary()
                store.close()
                self.assertEqual(before, after)

    def test_eval_selection_unknown_features_stay_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                store.close()

                result = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--json"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertIsNone(payload[0]["features"]["source"])
                self.assertIsNone(payload[0]["features"]["energy"])
                self.assertIsNone(payload[0]["features"]["bpm"])
                self.assertIsNone(payload[0]["features"]["vocalness"])

    def test_eval_selection_rejects_non_positive_limit(self) -> None:
        result = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--limit", "0"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--limit must be greater than zero", result.output)

    def test_eval_suite_json_outputs_prompts_candidates_and_red_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "loud-vocal.mp3", title="loud vocal", genre="pop"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        energy=0.91,
                        loudness=-5.0,
                        bpm=168.0,
                        vocalness=0.82,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                store.close()

                result = CliRunner().invoke(app, ["eval", "suite", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertGreaterEqual(len(payload), 4)
                first = payload[0]
                self.assertIn("prompt", first)
                self.assertIn("red_flag_count", first)
                self.assertEqual(first["candidates"][0]["features"]["source"], "model-essentia-voice-instrumental")
                self.assertIn("high vocalness in no-vocals top 3", first["candidates"][0]["red_flags"])

    def test_eval_suite_does_not_write_profile_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                before = store.profile_summary()
                store.close()

                result = CliRunner().invoke(app, ["eval", "suite", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                store = TonepathStore()
                after = store.profile_summary()
                store.close()
                self.assertEqual(before, after)

    def test_eval_suite_marks_unknown_features_as_low_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "unknown.mp3", title="unknown", genre=None))
                store.close()

                result = CliRunner().invoke(app, ["eval", "suite", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertIsNone(payload[0]["candidates"][0]["features"]["source"])
                self.assertIn("low evidence in top 3", payload[0]["candidates"][0]["red_flags"])

    def test_eval_suite_rejects_non_positive_limit(self) -> None:
        result = CliRunner().invoke(app, ["eval", "suite", "--limit", "0"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--limit must be greater than zero", result.output)


def track_for(path: Path, title: str, genre: str | None) -> Track:
    """Create one persisted-test track payload."""

    path.write_bytes(b"not decoded as audio")
    return Track(
        id=None,
        path=path,
        file_hash=path.name,
        mtime=1.0,
        title=title,
        artist="artist",
        album=None,
        genre=genre,
        duration=180.0,
        format="mp3",
    )


if __name__ == "__main__":
    unittest.main()
