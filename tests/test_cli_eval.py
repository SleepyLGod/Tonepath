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
                self.assertIn("yellow_flag_count", first)
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

    def test_eval_audit_json_outputs_evidence_pack(self) -> None:
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
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                before = store.profile_summary()
                store.close()

                result = CliRunner().invoke(
                    app,
                    ["eval", "audit", "我现在很烦，想半小时后进入写代码状态，不要人声", "--json", "--limit", "1"],
                )

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertIn("run_id", payload)
                self.assertTrue(Path(payload["evidence_path"]).exists())
                self.assertEqual(payload["candidates"][0]["track"]["title"], "quiet")
                self.assertEqual(payload["candidates"][0]["yellow_flags"], [])
                store = TonepathStore()
                after = store.profile_summary()
                store.close()
                self.assertEqual(before, after)

    def test_eval_audit_codex_missing_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                store.close()

                with patch("tonepath.evaluation.shutil.which", return_value=None):
                    result = CliRunner().invoke(app, ["eval", "audit", "focus 30m", "--codex"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("Codex CLI is not available", result.output)

    def test_eval_audit_codex_uses_read_only_sandbox_and_search_only_when_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            commands: list[list[str]] = []

            def fake_run(command: list[str], input: str, text: bool, check: bool) -> None:
                commands.append(command)
                output_path = Path(command[command.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "ok",
                            "decisions": [
                                {
                                    "track_id": 1,
                                    "decision": "keep",
                                    "fit_score": 0.9,
                                    "risk_flags": [],
                                    "reason": "local evidence fits",
                                    "evidence_used": [{"type": "local", "field": "bpm", "value": 84}],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                store.close()

                with patch("tonepath.evaluation.shutil.which", return_value="/usr/bin/codex"), patch(
                    "tonepath.evaluation.subprocess.run", side_effect=fake_run
                ):
                    result = CliRunner().invoke(app, ["eval", "audit", "focus 30m", "--codex", "--web", "--json"])
                    no_web_result = CliRunner().invoke(app, ["eval", "audit", "focus 30m", "--codex", "--json"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(no_web_result.exit_code, 0, no_web_result.output)
            self.assertIn("--search", commands[0])
            self.assertNotIn("--search", commands[1])
            self.assertIn("read-only", commands[0])
            payload = json.loads(result.output)
            self.assertEqual(payload["codex"]["decisions"][0]["decision"], "keep")

    def test_eval_audit_codex_rejects_web_evidence_without_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def fake_run(command: list[str], input: str, text: bool, check: bool) -> None:
                output_path = Path(command[command.index("-o") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "summary": "bad",
                            "decisions": [
                                {
                                    "track_id": 1,
                                    "decision": "demote",
                                    "fit_score": 0.2,
                                    "risk_flags": ["uncited web claim"],
                                    "reason": "web evidence lacks url",
                                    "evidence_used": [{"type": "web", "note": "uncited"}],
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                store.close()

                with patch("tonepath.evaluation.shutil.which", return_value="/usr/bin/codex"), patch(
                    "tonepath.evaluation.subprocess.run", side_effect=fake_run
                ):
                    result = CliRunner().invoke(app, ["eval", "audit", "focus 30m", "--codex"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Web evidence entries must include a URL", result.output)


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
