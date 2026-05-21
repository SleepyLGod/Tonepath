import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.evaluation import (
    annotate_red_flags,
    codex_audit_schema_path,
    codex_prompt,
    codex_skill_path,
    evaluate_audit,
    evaluate_intent,
    evaluate_rerank,
    evaluate_suite,
)
from tonepath.models import Track, TrackFeatures


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(output: str) -> str:
    return ANSI_RE.sub("", output)


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
        self.assertIn("--limit must be greater than zero", plain_output(result.output))

    def test_parse_outputs_low_stimulation_constraint(self) -> None:
        result = CliRunner().invoke(app, ["parse", "我要写论文，四十五分钟，低刺激，最好不要人声"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["constraints"], ["avoid_vocals", "low_stimulation"])

    def test_eval_audit_includes_low_stimulation_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "quiet.mp3", title="quiet", genre="ambient"))

                payload = evaluate_audit(store, "我要写论文，四十五分钟，低刺激，最好不要人声", limit=1)

                self.assertEqual(payload["constraints"], ["avoid_vocals", "low_stimulation"])
                store.close()

    def test_eval_suite_includes_low_stimulation_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "quiet.mp3", title="quiet", genre="ambient"))

                payload = evaluate_suite(store, limit=1, prompts=("我要写论文，四十五分钟，低刺激，最好不要人声",))

                self.assertEqual(payload[0]["constraints"], ["avoid_vocals", "low_stimulation"])
                store.close()

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
                self.assertIn("dirty_metadata_count", first)
                self.assertIn("duplicate_candidate_count", first)
                self.assertEqual(first["candidates"][0]["features"]["source"], "model-essentia-voice-instrumental")
                self.assertIn("high vocalness in no-vocals candidate", first["candidates"][0]["red_flags"])

    def test_eval_output_includes_clean_display_metadata_and_hygiene_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "clean-title.mp3", title="Song(null)", genre="ambient"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        energy=0.4,
                        loudness=-16.0,
                        bpm=92.0,
                        vocalness=0.2,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                store.close()

                result = CliRunner().invoke(app, ["eval", "suite", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                track = payload[0]["candidates"][0]["track"]
                self.assertEqual(track["display_title"], "Song")
                self.assertEqual(track["display_artist"], "artist")
                self.assertEqual(track["display_label"], "Song - artist")
                self.assertEqual(track["metadata_issues"], ["dirty title"])
                self.assertEqual(payload[0]["dirty_metadata_count"], 1)

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
                self.assertIn("low evidence candidate", payload[0]["candidates"][0]["red_flags"])

    def test_red_flags_apply_to_later_candidates(self) -> None:
        rows = [
            {
                "phase": "decompress",
                "confidence": "high",
                "features": {"source": "test", "energy": 0.2, "loudness": -18.0, "bpm": 80.0, "vocalness": 0.1},
            },
            {
                "phase": "stabilize",
                "confidence": "high",
                "features": {"source": "test", "energy": 0.4, "loudness": -14.0, "bpm": 90.0, "vocalness": 0.2},
            },
            {
                "phase": "focus",
                "confidence": "high",
                "features": {"source": "test", "energy": 0.5, "loudness": -12.0, "bpm": 100.0, "vocalness": 0.2},
            },
            {
                "phase": "focus",
                "confidence": "high",
                "features": {"source": "test", "energy": 0.91, "loudness": -5.0, "bpm": 168.0, "vocalness": 0.82},
            },
        ]

        annotate_red_flags(rows, no_vocals=True)

        self.assertIn("high vocalness in no-vocals candidate", rows[3]["red_flags"])
        self.assertIn("high energy in calm/focus candidate", rows[3]["red_flags"])
        self.assertNotIn("top 3", " ".join(rows[3]["red_flags"]))

    def test_eval_suite_rejects_non_positive_limit(self) -> None:
        result = CliRunner().invoke(app, ["eval", "suite", "--limit", "0"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--limit must be greater than zero", plain_output(result.output))

    def test_evaluate_intent_passes_packaged_corpus(self) -> None:
        payload = evaluate_intent()

        self.assertGreaterEqual(payload["total"], 50)
        self.assertEqual(payload["failed"], 0)
        self.assertEqual(payload["passed"], payload["total"])
        self.assertEqual(payload["failures"], [])

    def test_eval_intent_json_outputs_stable_summary(self) -> None:
        result = CliRunner().invoke(app, ["eval", "intent", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertGreaterEqual(payload["total"], 50)
        self.assertEqual(payload["failed"], 0)
        self.assertIn("cases", payload)
        self.assertIn("actual", payload["cases"][0])
        self.assertIn("expected", payload["cases"][0])

    def test_eval_intent_text_outputs_summary(self) -> None:
        result = CliRunner().invoke(app, ["eval", "intent"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Intent fixtures:", result.output)
        self.assertIn("failed 0", result.output)

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

    def test_codex_resources_resolve_from_package(self) -> None:
        skill_path = codex_skill_path()
        schema_path = codex_audit_schema_path()

        self.assertTrue(skill_path.exists())
        self.assertTrue(schema_path.exists())
        self.assertIn("src/tonepath/resources", str(skill_path))
        self.assertIn("src/tonepath/resources", str(schema_path))

    def test_codex_skill_documents_evidence_semantics_thresholds_and_examples(self) -> None:
        skill = codex_skill_path().read_text(encoding="utf-8")

        for expected in [
            "Evidence Field Semantics",
            "`score`",
            "`confidence`",
            "`features.energy`",
            "`features.loudness`",
            "`features.bpm`",
            "`features.vocalness`",
            "Threshold Guide",
            "`vocalness <= 0.35`",
            "`0.35..0.65`",
            "`>= 0.65`",
            "`BPM >= 140`",
            "`energy >= 0.68`",
            "`loudness >= -9.0`",
            "Keep example",
            "Demote example",
            "Reject example",
        ]:
            self.assertIn(expected, skill)

    def test_codex_prompt_points_to_skill_contract(self) -> None:
        prompt = codex_prompt("/tmp/evidence.json", web=True)

        self.assertIn("field semantics", prompt)
        self.assertIn("threshold guide", prompt)
        self.assertIn("examples", prompt)

    def test_eval_audit_codex_uses_read_only_sandbox_and_search_only_when_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            commands: list[list[str]] = []

            def fake_run(command: list[str], input: str, text: bool, check: bool, **kwargs: object) -> None:
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

            def fake_run(command: list[str], input: str, text: bool, check: bool, **kwargs: object) -> None:
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

    def test_eval_rerank_latest_uses_matching_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                write_audit_cache(Path(tmp) / "home", "run-1", "other prompt", [candidate(1, "wrong")], [])
                write_audit_cache(
                    Path(tmp) / "home",
                    "run-2",
                    "focus 30m",
                    [candidate(1, "quiet"), candidate(2, "loud")],
                    [decision(1, "keep"), decision(2, "reject", risk_flags=["too vocal"])],
                )

                result = CliRunner().invoke(app, ["eval", "rerank", "focus 30m", "--latest", "--json"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertTrue(payload["found"])
                self.assertEqual(payload["run_id"], "run-2")
                self.assertEqual(payload["counts"]["keep"], 1)
                self.assertEqual(payload["counts"]["reject"], 1)
                self.assertEqual([row["track"]["title"] for row in payload["suggested_queue"]], ["quiet"])

    def test_eval_rerank_latest_ignores_stale_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                write_audit_cache(
                    Path(tmp) / "home",
                    "run-1",
                    "evening relaxation",
                    [candidate(1, "quiet")],
                    [decision(1, "keep")],
                )

                result = CliRunner().invoke(app, ["eval", "rerank", "focus 30m", "--latest"])

                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn("No matching Codex audit result", result.output)

    def test_eval_rerank_rules_keep_demote_reject_and_missing_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                write_audit_cache(
                    Path(tmp) / "home",
                    "run-1",
                    "focus 30m",
                    [
                        candidate(1, "keep me"),
                        candidate(2, "missing"),
                        candidate(3, "demote me"),
                        candidate(4, "reject me"),
                    ],
                    [decision(1, "keep"), decision(3, "demote"), decision(4, "reject")],
                )

                payload = evaluate_rerank("focus 30m")

                self.assertTrue(payload["found"])
                self.assertEqual([row["decision"] for row in payload["details"]], ["keep", "not_audited", "demote", "reject"])
                self.assertEqual(
                    [row["track"]["title"] for row in payload["suggested_queue"]],
                    ["keep me", "missing", "demote me"],
                )
                rejected = payload["details"][3]
                self.assertEqual(rejected["suggested_action"], "remove from suggested queue")
                self.assertNotIn(rejected, payload["suggested_queue"])

    def test_eval_rerank_does_not_write_profile_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                write_audit_cache(
                    Path(tmp) / "home",
                    "run-1",
                    "focus 30m",
                    [candidate(1, "quiet")],
                    [decision(1, "keep")],
                )
                store = TonepathStore()
                before = store.profile_summary()
                store.close()

                result = CliRunner().invoke(app, ["eval", "rerank", "focus 30m", "--latest"])

                self.assertEqual(result.exit_code, 0, result.output)
                store = TonepathStore()
                after = store.profile_summary()
                store.close()
                self.assertEqual(before, after)


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


def write_audit_cache(
    home: Path,
    run_id: str,
    prompt: str,
    candidates: list[dict[str, object]],
    decisions: list[dict[str, object]],
) -> None:
    """Write one local Codex audit cache fixture."""

    result_dir = home / "cache" / "audit" / run_id
    result_dir.mkdir(parents=True)
    (result_dir / "evidence.json").write_text(
        json.dumps({"run_id": run_id, "prompt": prompt, "candidates": candidates}),
        encoding="utf-8",
    )
    (result_dir / "codex-result.json").write_text(
        json.dumps({"summary": f"summary {run_id}", "decisions": decisions}),
        encoding="utf-8",
    )


def candidate(track_id: int, title: str) -> dict[str, object]:
    """Return a minimal audit candidate fixture."""

    return {
        "phase": "focus",
        "track": {"id": track_id, "title": title, "artist": "artist"},
        "score": 1.0,
        "confidence": "high",
        "features": {},
        "reasons": [],
        "red_flags": [],
        "yellow_flags": [],
    }


def decision(track_id: int, value: str, risk_flags: list[str] | None = None) -> dict[str, object]:
    """Return a valid Codex audit decision fixture."""

    return {
        "track_id": track_id,
        "decision": value,
        "fit_score": 0.8,
        "risk_flags": risk_flags or [],
        "reason": f"{value} reason",
        "evidence_used": [{"type": "local", "field": "bpm", "value": 90}],
    }


if __name__ == "__main__":
    unittest.main()
