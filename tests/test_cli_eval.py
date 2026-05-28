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
    clap_probe_for_phase,
    codex_audit_schema_path,
    codex_prompt,
    codex_skill_path,
    diagnose_bakeoff_payload,
    evaluate_audit,
    evaluate_bakeoff,
    evaluate_diagnose,
    evaluate_intent,
    evaluate_rerank,
    evaluate_suite,
    hybrid_candidates,
    profile_movements,
)
from tonepath.models import ProfileRule, SessionPhase, SessionPlan, SessionRequest, Track, TrackFeatures
from tonepath.planner import plan_session


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
                self.assertTrue(payload["profile_enabled"])
                row = payload["candidates"][0]
                self.assertEqual(row["phase"], "decompress")
                self.assertEqual(row["track"]["title"], "quiet")
                self.assertEqual(row["features"]["source"], "model-audio-separator")
                self.assertEqual(row["features"]["confidence"], "high")
                self.assertEqual(row["features"]["vocalness"], 0.18)
                self.assertIn("vocalness feature supports no-vocals constraint", row["reasons"])

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

    def test_eval_bakeoff_json_outputs_engine_results(self) -> None:
        scenario = {
            "id": "sad_test",
            "lang": "zh",
            "prompt": "我有点难过，想慢慢开心一点，但不要太吵",
            "limit": 2,
            "expected_intent": {"source_state": "low", "target_state": "uplift", "duration_min": 30, "constraints": ["low_stimulation", "gentle_uplift"]},
            "checks": [{"type": "no_duplicate_candidates", "level": "fail"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                warm_id = store.upsert_track(track_for(Path(tmp) / "warm.mp3", title="warm", genre="ambient"))
                dark_id = store.upsert_track(track_for(Path(tmp) / "dark.mp3", title="dark", genre="soundtrack"))
                store.upsert_features(TrackFeatures(warm_id, energy=0.3, loudness=-14.0, bpm=90.0, valence_estimate=0.7, feature_source="test", confidence="high"))
                store.upsert_features(TrackFeatures(dark_id, energy=0.4, loudness=-12.0, bpm=100.0, valence_estimate=0.4, feature_source="test", confidence="high"))
                store.close()

                def audio_embedding(track: Track) -> list[float] | None:
                    return [1.0, 0.0] if track.title == "warm" else [0.0, 1.0]

                with patch("tonepath.evaluation.load_benchmark_scenarios", return_value=[scenario]), patch(
                    "tonepath.evaluation.read_or_create_clap_text_embeddings", return_value={}
                ), patch(
                    "tonepath.evaluation.read_or_create_clap_text_embedding", return_value=[1.0, 0.0]
                ), patch("tonepath.evaluation.read_clap_audio_embedding", side_effect=audio_embedding):
                    result = CliRunner().invoke(app, ["eval", "bakeoff", "--engine", "selector", "--engine", "clap", "--engine", "hybrid", "--limit", "2", "--json"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload[0]["scenario_id"], "sad_test")
                self.assertEqual([engine["engine"] for engine in payload[0]["engines"]], ["selector", "clap", "hybrid"])
                self.assertEqual([row["compared_engine"] for row in payload[0]["deltas"]], ["clap", "hybrid"])
                self.assertEqual(payload[0]["engines"][1]["candidates"][0]["track"]["title"], "warm")

    def test_eval_bakeoff_marks_missing_clap_embeddings_as_fail_or_warning(self) -> None:
        scenario = {
            "id": "sad_test",
            "lang": "zh",
            "prompt": "我有点难过，想慢慢开心一点，但不要太吵",
            "limit": 1,
            "expected_intent": {},
            "checks": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "warm.mp3", title="warm", genre="ambient"))
                before = store.profile_summary()
                with patch("tonepath.evaluation.load_benchmark_scenarios", return_value=[scenario]), patch(
                    "tonepath.evaluation.read_or_create_clap_text_embeddings", return_value={}
                ), patch(
                    "tonepath.evaluation.read_or_create_clap_text_embedding", return_value=[1.0]
                ), patch("tonepath.evaluation.read_clap_audio_embedding", return_value=None):
                    payload = evaluate_bakeoff(store, ("selector", "clap", "hybrid"), 1)
                after = store.profile_summary()
                store.close()

                clap = payload[0]["engines"][1]
                hybrid = payload[0]["engines"][2]
                self.assertEqual(clap["result"], "FAIL")
                self.assertEqual(hybrid["result"], "WARN")
                self.assertEqual(hybrid["candidates"][0]["track"]["title"], "warm")
                self.assertEqual(before, after)

    def test_eval_bakeoff_rejects_unknown_engine(self) -> None:
        result = CliRunner().invoke(app, ["eval", "bakeoff", "--engine", "selector", "--engine", "bogus"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unsupported bake-off engine", result.output)

    def test_eval_diagnose_json_outputs_root_causes(self) -> None:
        payload = {
            "summary": {
                "scenario_count": 1,
                "result_counts": {"PASS": 0, "WARN": 1, "FAIL": 0},
                "root_cause_counts": {"metadata_hygiene": 1},
                "top_root_causes": [{"cause": "metadata_hygiene", "count": 1}],
                "recommended_next_action": "Clean track metadata so recommendation output is more trustworthy.",
            },
            "scenarios": [
                {
                    "scenario_id": "metadata",
                    "prompt": "focus",
                    "overall_result": "WARN",
                    "engines": [],
                    "issues": [],
                    "root_causes": ["metadata_hygiene"],
                    "next_action": "Clean track metadata so recommendation output is more trustworthy.",
                }
            ],
        }
        with patch("tonepath.cli.evaluate_diagnose", return_value=payload):
            result = CliRunner().invoke(app, ["eval", "diagnose", "--json", "--limit", "1"])

        self.assertEqual(result.exit_code, 0, result.output)
        parsed = json.loads(result.output)
        self.assertEqual(parsed["scenarios"][0]["root_causes"], ["metadata_hygiene"])

    def test_eval_diagnose_does_not_write_profile_state(self) -> None:
        scenario = {
            "id": "metadata",
            "lang": "en",
            "prompt": "focus 30m",
            "limit": 1,
            "expected_intent": {},
            "checks": [{"type": "metadata_hygiene_warning", "level": "warn"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "song.mp3", title="Song(null)", genre="ambient"))
                store.upsert_features(TrackFeatures(track_id=track_id, energy=0.2, loudness=-18.0, bpm=80.0, feature_source="test", confidence="high"))
                before = store.profile_summary()
                with patch("tonepath.evaluation.load_benchmark_scenarios", return_value=[scenario]), patch(
                    "tonepath.evaluation.read_or_create_clap_text_embeddings", return_value={}
                ), patch("tonepath.evaluation.read_or_create_clap_text_embedding", return_value=[1.0]), patch(
                    "tonepath.evaluation.read_clap_audio_embedding", return_value=None
                ):
                    payload = evaluate_diagnose(store, limit=1)
                after = store.profile_summary()
                store.close()

                self.assertEqual(before, after)
                self.assertIn("metadata_hygiene", payload["scenarios"][0]["root_causes"])

    def test_diagnose_classifies_clap_regression(self) -> None:
        payload = diagnose_bakeoff_payload(
            [
                diagnose_scenario_payload(
                    [
                        diagnose_engine("selector", "PASS", []),
                        diagnose_engine("clap", "FAIL", []),
                        diagnose_engine("hybrid", "PASS", []),
                    ],
                    deltas=[
                        {"baseline_engine": "selector", "compared_engine": "clap", "verdict": "regressed"},
                        {"baseline_engine": "selector", "compared_engine": "hybrid", "verdict": "inconclusive"},
                    ],
                )
            ]
        )

        causes = payload["scenarios"][0]["root_causes"]
        self.assertIn("clap_regression", causes)
        self.assertIn("hybrid_inconclusive", causes)

    def test_diagnose_classifies_selector_tuning_with_available_features(self) -> None:
        payload = diagnose_bakeoff_payload(
            [
                diagnose_scenario_payload(
                    [
                        diagnose_engine(
                            "selector",
                            "FAIL",
                            [diagnose_check("max_stimulation_top_k", "fail", [1])],
                            candidates=[diagnose_candidate("loud", energy=0.8, loudness=-6.0, bpm=150.0, vocalness=0.2)],
                        ),
                        diagnose_engine("clap", "FAIL", []),
                        diagnose_engine("hybrid", "FAIL", []),
                    ]
                )
            ]
        )

        self.assertIn("selector_tuning", payload["scenarios"][0]["root_causes"])

    def test_diagnose_classifies_missing_model_evidence(self) -> None:
        payload = diagnose_bakeoff_payload(
            [
                diagnose_scenario_payload(
                    [
                        diagnose_engine(
                            "selector",
                            "FAIL",
                            [diagnose_check("required_affect_top_k", "fail", [1])],
                            candidates=[diagnose_candidate("unknown", source=None, valence=None, affect_profile={})],
                        ),
                        diagnose_engine("clap", "FAIL", []),
                        diagnose_engine("hybrid", "FAIL", []),
                    ]
                )
            ]
        )

        self.assertIn("model_evidence_weak", payload["scenarios"][0]["root_causes"])

    def test_diagnose_classifies_benchmark_threshold_when_all_engines_fail_cleanly(self) -> None:
        payload = diagnose_bakeoff_payload(
            [
                diagnose_scenario_payload(
                    [
                        diagnose_engine(
                            "selector",
                            "FAIL",
                            [diagnose_check("custom_threshold", "fail", [1])],
                            candidates=[diagnose_candidate("clean")],
                        ),
                        diagnose_engine("clap", "FAIL", []),
                        diagnose_engine("hybrid", "FAIL", []),
                    ]
                )
            ]
        )

        self.assertIn("benchmark_threshold", payload["scenarios"][0]["root_causes"])

    def test_hybrid_keeps_clap_inside_selector_safe_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                safe_id = store.upsert_track(track_for(Path(tmp) / "safe.mp3", title="safe", genre="instrumental"))
                risky_id = store.upsert_track(track_for(Path(tmp) / "risky.mp3", title="risky", genre="pop"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=safe_id,
                        energy=0.32,
                        loudness=-18.0,
                        bpm=84.0,
                        vocalness=0.1,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                store.upsert_features(
                    TrackFeatures(
                        track_id=risky_id,
                        energy=0.9,
                        loudness=-5.0,
                        bpm=168.0,
                        vocalness=0.92,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                plan = one_phase_plan(no_vocals=True)

                def audio_embedding(track: Track) -> list[float] | None:
                    return [1.0, 0.0] if track.title == "risky" else [0.0, 1.0]

                with patch("tonepath.evaluation.read_or_create_clap_text_embedding", return_value=[1.0, 0.0]), patch(
                    "tonepath.evaluation.read_clap_audio_embedding", side_effect=audio_embedding
                ):
                    candidates = hybrid_candidates(store, plan, 1)
                store.close()

                self.assertEqual(candidates[0].track.title, "safe")
                self.assertTrue(any("CLAP semantic bonus" in reason for reason in candidates[0].reasons))

    def test_hybrid_does_not_pull_tracks_outside_selector_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                for index in range(12):
                    track_id = store.upsert_track(track_for(Path(tmp) / f"pool-{index}.mp3", title=f"pool {index}", genre="ambient"))
                    store.upsert_features(
                        TrackFeatures(
                            track_id=track_id,
                            energy=0.28,
                            loudness=-18.0,
                            bpm=80.0 + index,
                            vocalness=0.1,
                            feature_source="model-essentia-voice-instrumental",
                            confidence="high",
                        )
                    )
                outsider_id = store.upsert_track(track_for(Path(tmp) / "outsider.mp3", title="outsider", genre="pop"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=outsider_id,
                        energy=0.95,
                        loudness=-4.0,
                        bpm=180.0,
                        vocalness=0.95,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                plan = one_phase_plan(no_vocals=True)

                def audio_embedding(track: Track) -> list[float] | None:
                    return [1.0, 0.0] if track.title == "outsider" else [0.0, 1.0]

                with patch("tonepath.evaluation.read_or_create_clap_text_embedding", return_value=[1.0, 0.0]), patch(
                    "tonepath.evaluation.read_clap_audio_embedding", side_effect=audio_embedding
                ):
                    candidates = hybrid_candidates(store, plan, 3)
                store.close()

                self.assertNotIn("outsider", [candidate.track.title for candidate in candidates])

    def test_chinese_bakeoff_prompt_uses_english_canonical_probe(self) -> None:
        plan = plan_session("我有点难过，想慢慢开心一点，但不要太吵")

        probe = clap_probe_for_phase(plan, "lift")

        self.assertIn("gentle uplifting warm calm music", probe)
        self.assertIn("hopeful, brighter", probe)
        self.assertIn("low stimulation", probe)

    def test_eval_selection_unknown_features_stay_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "song.mp3", title="song", genre=None))
                store.close()

                result = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--json"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                row = payload["candidates"][0]
                self.assertIsNone(row["features"]["source"])
                self.assertIsNone(row["features"]["energy"])
                self.assertIsNone(row["features"]["bpm"])
                self.assertIsNone(row["features"]["vocalness"])

    def test_eval_selection_can_disable_profile_rules(self) -> None:
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
                store.upsert_profile_rule(
                    ProfileRule(
                        id=None,
                        key="global:prefer_lower_loudness:loudness",
                        value=json.dumps(
                            {
                                "scope": "global",
                                "rule_type": "prefer_lower_loudness",
                                "target": "loudness",
                                "threshold": -12.0,
                                "weight": 0.7,
                                "confidence": "medium",
                                "source": "test",
                                "rationale": "focus prefers quieter tracks",
                                "evidence_count": 1,
                            }
                        ),
                        source="test",
                        confidence="medium",
                    )
                )
                store.close()

                with_profile = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--json", "--with-profile"])
                no_profile = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--json", "--no-profile"])

                self.assertEqual(with_profile.exit_code, 0, with_profile.output)
                self.assertEqual(no_profile.exit_code, 0, no_profile.output)
                with_payload = json.loads(with_profile.output)
                no_payload = json.loads(no_profile.output)
                self.assertTrue(with_payload["profile_enabled"])
                self.assertFalse(no_payload["profile_enabled"])
                self.assertNotEqual(with_payload["candidates"][0]["score"], no_payload["candidates"][0]["score"])
                self.assertTrue(any("profile rule:" in reason for reason in with_payload["candidates"][0]["reasons"]))
                self.assertFalse(any("profile rule:" in reason for reason in no_payload["candidates"][0]["reasons"]))

    def test_eval_selection_rejects_conflicting_profile_flags(self) -> None:
        result = CliRunner().invoke(app, ["eval", "selection", "focus 30m", "--with-profile", "--no-profile"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Choose only one", result.output)

    def test_eval_profile_reports_no_active_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                store.upsert_track(track_for(Path(tmp) / "quiet.mp3", title="quiet", genre="ambient"))
                before = store.profile_summary()
                store.close()

                result = CliRunner().invoke(app, ["eval", "profile", "focus 30m", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload["active_rule_count"], 0)
                self.assertIn("No active profile rules", payload["message"])
                self.assertFalse(payload["no_profile"]["profile_enabled"])
                self.assertTrue(payload["with_profile"]["profile_enabled"])
                store = TonepathStore()
                after = store.profile_summary()
                store.close()
                self.assertEqual(before, after)

    def test_eval_profile_reports_score_delta_and_profile_reason(self) -> None:
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
                store.upsert_profile_rule(
                    ProfileRule(
                        id=None,
                        key="global:prefer_lower_loudness:loudness",
                        value=json.dumps(
                            {
                                "scope": "global",
                                "rule_type": "prefer_lower_loudness",
                                "target": "loudness",
                                "threshold": -12.0,
                                "weight": 0.7,
                                "confidence": "medium",
                                "source": "test",
                                "rationale": "focus prefers quieter tracks",
                                "evidence_count": 1,
                            }
                        ),
                        source="test",
                        confidence="medium",
                    )
                )
                store.close()

                result = CliRunner().invoke(app, ["eval", "profile", "focus 30m", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload["active_rule_count"], 1)
                movement = payload["movements"][0]
                self.assertNotEqual(movement["score_delta"], 0)
                self.assertTrue(any("profile rule:" in reason for reason in movement["profile_reasons"]))

    def test_eval_profile_warns_when_lower_vocalness_lacks_high_bpm_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "fast.mp3", title="fast instrumental", genre="instrumental"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        energy=0.48,
                        loudness=-13.0,
                        bpm=144.0,
                        vocalness=0.12,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                store.upsert_profile_rule(profile_rule("global", "prefer_lower_vocalness", "vocalness", 0.35, 0.6))
                store.close()

                result = CliRunner().invoke(app, ["eval", "profile", "focus 30m no vocals", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload["warnings"], ["Lower-vocalness rule may need a high-BPM demotion companion."])

    def test_eval_profile_warns_when_small_limit_hides_high_bpm_companion_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                quiet_id = store.upsert_track(track_for(Path(tmp) / "quiet.mp3", title="quiet", genre="instrumental"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=quiet_id,
                        energy=0.24,
                        loudness=-18.2,
                        bpm=84.0,
                        vocalness=0.18,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                fast_id = store.upsert_track(track_for(Path(tmp) / "fast.mp3", title="fast instrumental", genre="instrumental"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=fast_id,
                        energy=0.48,
                        loudness=-13.0,
                        bpm=144.0,
                        vocalness=0.12,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                store.upsert_profile_rule(profile_rule("global", "prefer_lower_vocalness", "vocalness", 0.35, 0.6))
                store.close()

                result = CliRunner().invoke(app, ["eval", "profile", "focus 30m no vocals", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(
                    payload["warnings"],
                    ["Profile risk may be hidden by --limit; rerun with --limit 20 to check high-BPM companion risk."],
                )
                self.assertEqual(len(payload["with_profile"]["candidates"]), 1)

    def test_eval_profile_warning_disappears_with_high_bpm_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "fast.mp3", title="fast instrumental", genre="instrumental"))
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        energy=0.48,
                        loudness=-13.0,
                        bpm=144.0,
                        vocalness=0.12,
                        feature_source="model-essentia-voice-instrumental",
                        confidence="high",
                    )
                )
                store.upsert_profile_rule(profile_rule("global", "prefer_lower_vocalness", "vocalness", 0.35, 0.6))
                store.upsert_profile_rule(profile_rule("global", "demote_high_bpm", "bpm", 135.0, 0.8))
                store.close()

                result = CliRunner().invoke(app, ["eval", "profile", "focus 30m no vocals", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertEqual(payload["warnings"], [])

    def test_eval_profile_explains_when_active_rules_do_not_match_candidates(self) -> None:
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
                store.upsert_profile_rule(
                    ProfileRule(
                        id=None,
                        key="focus:prefer_lower_loudness:loudness",
                        value=json.dumps(
                            {
                                "scope": "focus",
                                "rule_type": "prefer_lower_loudness",
                                "target": "loudness",
                                "threshold": -12.0,
                                "weight": 0.7,
                                "confidence": "medium",
                                "source": "test",
                                "rationale": "focus prefers quieter tracks",
                                "evidence_count": 1,
                            }
                        ),
                        source="test",
                        confidence="medium",
                    )
                )
                store.close()

                result = CliRunner().invoke(app, ["eval", "profile", "focus 30m", "--json", "--limit", "1"])

                self.assertEqual(result.exit_code, 0, result.output)
                payload = json.loads(result.output)
                self.assertIn("did not match", payload["message"])
                self.assertIn("--limit", payload["message"])

    def test_profile_movements_consumes_duplicate_identities_and_ignores_empty_identity(self) -> None:
        rows = profile_movements(
            [
                {"track": {"display_label": "dup"}, "score": 1.0},
                {"track": {"display_label": "dup"}, "score": 2.0},
                {"track": {}, "score": 9.0},
            ],
            [
                {"track": {"display_label": "dup"}, "score": 3.0, "reasons": []},
                {"track": {"display_label": "dup"}, "score": 4.0, "reasons": []},
                {"track": {}, "score": 5.0, "reasons": []},
            ],
        )

        self.assertEqual(rows[0]["rank_no_profile"], 1)
        self.assertEqual(rows[0]["score_delta"], 2.0)
        self.assertEqual(rows[1]["rank_no_profile"], 2)
        self.assertEqual(rows[1]["score_delta"], 2.0)
        self.assertIsNone(rows[2]["rank_no_profile"])
        self.assertIsNone(rows[2]["score_delta"])

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


def one_phase_plan(no_vocals: bool = False) -> SessionPlan:
    """Return a one-phase plan for focused bake-off tests."""

    return SessionPlan(
        request=SessionRequest(
            prompt="focus",
            source_state="unspecified",
            target_state="focus",
            duration_sec=1800,
            no_vocals=no_vocals,
            quiet=True,
        ),
        phases=(
            SessionPhase(
                label="focus",
                start_sec=0,
                end_sec=1800,
                target_arousal=0.35,
                target_valence=0.55,
                target_energy=0.35,
                vocal_policy="avoid" if no_vocals else "allow",
            ),
        ),
    )


def diagnose_scenario_payload(
    engines: list[dict[str, object]],
    deltas: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return a minimal bake-off scenario payload for diagnosis tests."""

    return {
        "scenario_id": "scenario",
        "lang": "en",
        "prompt": "prompt",
        "limit": 1,
        "engines": engines,
        "deltas": deltas or [],
    }


def diagnose_engine(
    name: str,
    result: str,
    checks: list[dict[str, object]],
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return a minimal bake-off engine payload for diagnosis tests."""

    return {
        "engine": name,
        "result": result,
        "checks": checks,
        "red_flag_count": 0,
        "yellow_flag_count": 0,
        "candidates": candidates or [],
    }


def diagnose_check(check_type: str, status: str, affected_ranks: list[int]) -> dict[str, object]:
    """Return a minimal benchmark check payload for diagnosis tests."""

    return {
        "type": check_type,
        "status": status,
        "message": f"{check_type} {status}",
        "affected_ranks": affected_ranks,
    }


def diagnose_candidate(
    title: str,
    source: str | None = "test",
    energy: float | None = 0.3,
    loudness: float | None = -14.0,
    bpm: float | None = 90.0,
    vocalness: float | None = 0.2,
    valence: float | None = 0.6,
    affect_profile: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return a minimal evaluation candidate payload for diagnosis tests."""

    return {
        "phase": "focus",
        "track": {"display_label": f"{title} - artist", "title": title},
        "confidence": "high" if source else "low",
        "features": {
            "source": source,
            "confidence": "high" if source else "low",
            "energy": energy,
            "loudness": loudness,
            "bpm": bpm,
            "vocalness": vocalness,
            "valence": valence,
            "affect_profile": {"uplift": 0.4} if affect_profile is None else affect_profile,
        },
        "red_flags": [],
        "yellow_flags": [],
    }


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


def profile_rule(scope: str, rule_type: str, target: str, threshold: float, weight: float) -> ProfileRule:
    """Return one active profile rule fixture."""

    return ProfileRule(
        id=None,
        key=f"{scope}:{rule_type}:{target}",
        value=json.dumps(
            {
                "scope": scope,
                "rule_type": rule_type,
                "target": target,
                "threshold": threshold,
                "weight": weight,
                "confidence": "medium",
                "source": "test",
                "rationale": f"{scope} {rule_type}",
                "evidence_count": 1,
            }
        ),
        source="test",
        confidence="medium",
    )


if __name__ == "__main__":
    unittest.main()
