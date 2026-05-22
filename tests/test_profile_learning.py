import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.planner import plan_session
from tonepath.profile import build_profile_evidence, deterministic_suggestions
from tonepath.selector import score_track


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
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                result = CliRunner().invoke(app, ["profile", "suggest", "--llm"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("requires --confirm", result.output)

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
