import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.planner import plan_session
from tonepath.profile import build_profile_evidence, deterministic_suggestions, suggest_with_llm
from tonepath.selector import score_track


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(output: str) -> str:
    return ANSI_RE.sub("", output)


class ProfileLearningTest(unittest.TestCase):
    def test_profile_evidence_omits_paths_and_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)

                evidence = build_profile_evidence(store)
                text = json.dumps(evidence, ensure_ascii=False)

                self.assertNotIn(str(Path(tmp)), text)
                self.assertNotIn("secret", text)
                self.assertIn("too-loud", text)
                self.assertIn("quiet song - artist", text)
                store.close()

    def test_deterministic_suggestions_use_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                store.record_feedback("like", session_id=session_id, track_id=track_id)

                suggestions = deterministic_suggestions(build_profile_evidence(store))

                self.assertIn("prefer_lower_loudness", {item["rule_type"] for item in suggestions})
                self.assertIn("prefer_lower_vocalness", {item["rule_type"] for item in suggestions})
                store.close()

    def test_profile_suggest_llm_requires_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                result = CliRunner().invoke(app, ["profile", "suggest", "--llm"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("requires --confirm", plain_output(result.output))
                result = CliRunner().invoke(app, ["profile", "suggest", "--llm", "--memory"])
                self.assertNotEqual(result.exit_code, 0)
                self.assertFalse((home / "profile" / "memory.md").exists())

    def test_profile_suggest_llm_writes_pending_suggestions(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "suggestions": [
                                    {
                                        "suggestion_id": "focus-lower-loudness",
                                        "scope": "focus",
                                        "rule_type": "prefer_lower_loudness",
                                        "target": "loudness",
                                        "threshold": -12.0,
                                        "weight": 0.7,
                                        "confidence": "medium",
                                        "rationale": "The user marked focus tracks as too loud.",
                                        "evidence_count": 1,
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                store.close()
                with patch("tonepath.profile.urllib.request.urlopen", return_value=response):
                    result = CliRunner().invoke(app, ["profile", "suggest", "--llm", "--confirm"])

                self.assertEqual(result.exit_code, 0, result.output)
                suggestion_paths = list((Path(tmp) / "home" / "cache" / "profile").glob("*/suggestions.json"))
                self.assertTrue(suggestion_paths)
                self.assertIn("focus-lower-loudness", suggestion_paths[0].read_text(encoding="utf-8"))

    def test_profile_memory_write_preserves_human_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                store.close()

                result = CliRunner().invoke(app, ["profile", "memory", "write"])
                self.assertEqual(result.exit_code, 0, result.output)
                memory_path = home / "profile" / "memory.md"
                text = memory_path.read_text(encoding="utf-8")
                edited = text.replace(
                    "Add personal listening notes here. Tonepath preserves this section when regenerating the file.",
                    "I prefer focus music that stays in the background.",
                )
                memory_path.write_text(edited, encoding="utf-8")

                result = CliRunner().invoke(app, ["profile", "memory", "write"])
                self.assertEqual(result.exit_code, 0, result.output)

                updated = memory_path.read_text(encoding="utf-8")
                self.assertIn("I prefer focus music that stays in the background.", updated)
                self.assertIn("too-loud", updated)
                self.assertNotIn(str(Path(tmp)), updated)

    def test_profile_memory_write_handles_malformed_human_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                memory_path = home / "profile" / "memory.md"
                memory_path.parent.mkdir(parents=True)
                memory_path.write_text("<!-- tonepath:human-notes:start -->\nunfinished note", encoding="utf-8")

                result = CliRunner().invoke(app, ["profile", "memory", "write"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Tonepath preserves this section", memory_path.read_text(encoding="utf-8"))

    def test_profile_evidence_write_creates_markdown_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                store.close()

                result = CliRunner().invoke(app, ["profile", "evidence", "write"])

                self.assertEqual(result.exit_code, 0, result.output)
                latest = home / "profile" / "evidence" / "latest.md"
                text = latest.read_text(encoding="utf-8")
                self.assertIn("Tonepath Profile Evidence", text)
                self.assertIn("too-loud", text)
                self.assertIn("quiet song - artist", text)
                self.assertNotIn(str(Path(tmp)), text)
                self.assertNotIn("secret", text)

    def test_profile_suggest_llm_with_memory_sends_markdown_context(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "suggestions": [
                                    {
                                        "suggestion_id": "focus-lower-loudness",
                                        "scope": "focus",
                                        "rule_type": "prefer_lower_loudness",
                                        "target": "loudness",
                                        "threshold": -12.0,
                                        "weight": 0.7,
                                        "confidence": "medium",
                                        "rationale": "The user marked focus tracks as too loud.",
                                        "evidence_count": 1,
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                store.close()
                with patch("tonepath.profile.urllib.request.urlopen", return_value=response) as mocked:
                    result = CliRunner().invoke(app, ["profile", "suggest", "--llm", "--confirm", "--memory"])

                self.assertEqual(result.exit_code, 0, result.output)
                request = mocked.call_args.args[0]
                payload = json.loads(request.data.decode("utf-8"))
                user_message = json.loads(payload["messages"][1]["content"])
                self.assertIn("profile_memory_markdown", user_message)
                self.assertIn("Tonepath Profile Memory", user_message["profile_memory_markdown"])
                self.assertIn("Tonepath Profile Evidence", user_message["profile_memory_markdown"])
                self.assertTrue((home / "profile" / "memory.md").exists())
                self.assertTrue((home / "profile" / "evidence" / "latest.md").exists())

    def test_llm_profile_prompt_contains_rule_guidance(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": "{\"suggestions\": []}"}}]}).encode("utf-8")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"}, clear=True):
            with patch("tonepath.profile.urllib.request.urlopen", return_value=response) as mocked:
                suggestions = suggest_with_llm({"run_id": "r1", "feedback_events": []}, memory_context="# Tonepath Profile Memory")

        self.assertEqual(suggestions, [])
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("too-loud feedback and known loudness", system_prompt)
        self.assertIn("liked low-vocalness track", system_prompt)
        self.assertIn("BPM >= 135", system_prompt)
        self.assertIn("Supported rule_type values", system_prompt)

    def test_invalid_llm_suggestion_is_rejected(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "suggestions": [
                                        {
                                            "suggestion_id": "bad",
                                            "scope": "focus",
                                            "rule_type": "invent_music_fact",
                                            "target": "genre",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret"}, clear=True):
            with patch("tonepath.profile.urllib.request.urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "Unsupported profile rule type"):
                    suggest_with_llm({"run_id": "r1", "feedback_events": []})

    def test_profile_suggest_llm_invalid_suggestion_has_clear_error(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "suggestions": [
                                        {
                                            "suggestion_id": "bad",
                                            "scope": "focus",
                                            "rule_type": "",
                                            "target": "loudness",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                store.close()
                with patch("tonepath.profile.urllib.request.urlopen", return_value=response):
                    result = CliRunner().invoke(app, ["profile", "suggest", "--llm", "--confirm", "--memory"])

                self.assertNotEqual(result.exit_code, 0)
                output = plain_output(result.output)
                self.assertIn("Unsupported profile rule type", output)
                self.assertNotIn("Traceback", output)

    def test_profile_apply_affects_selector_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store, track_id = populated_store(tmp, loudness=-8.0, bpm=120.0, vocalness=0.2)
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                evidence = build_profile_evidence(store)
                suggestions = deterministic_suggestions(evidence)
                from tonepath.profile import save_suggestions

                save_suggestions(evidence, suggestions, "deterministic")
                result = CliRunner().invoke(app, ["profile", "apply", "focus-lower-loudness"])
                self.assertEqual(result.exit_code, 0, result.output)

                track = store.list_tracks()[0]
                phase = plan_session("focus 30m").phases[-1]
                candidate = score_track(store, track, phase)

                self.assertTrue(any("profile rule:" in reason for reason in candidate.reasons))
                self.assertEqual(store.profile_summary()["profile_rules"], 1)
                store.close()

    def test_roadmap_documents_profile_comparison_loop(self) -> None:
        roadmap = (Path(__file__).resolve().parents[1] / "docs" / "tonepath-private-radio-agent-roadmap.md").read_text(encoding="utf-8")

        self.assertIn("--no-profile", roadmap)
        self.assertIn("--with-profile", roadmap)
        self.assertIn("personalized radio loop is still early-stage", roadmap)


def populated_store(tmp: str, loudness: float, bpm: float, vocalness: float) -> tuple[TonepathStore, int]:
    store = TonepathStore()
    path = Path(tmp) / "quiet.mp3"
    path.write_bytes(b"fake")
    track_id = store.upsert_track(
        Track(
            id=None,
            path=path,
            file_hash="hash",
            mtime=1.0,
            title="quiet song",
            artist="artist",
            album=None,
            genre="ambient",
            duration=180.0,
            format="mp3",
        )
    )
    store.upsert_features(
        TrackFeatures(
            track_id=track_id,
            bpm=bpm,
            loudness=loudness,
            energy=0.4,
            vocalness=vocalness,
            feature_source="model-essentia-voice-instrumental",
            confidence="high",
        )
    )
    return store, track_id
