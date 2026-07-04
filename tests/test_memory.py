"""Tests for private memory logs, profile consolidation, and suggestions."""

import json
import os
import re
import stat
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.memory import add_memory_log, build_memory_evidence, consolidate_memory_with_llm, memory_log_path, memory_profile_path, read_memory_logs


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_output(output: str) -> str:
    return ANSI_RE.sub("", output)


class MemoryCliTest(unittest.TestCase):
    def test_memory_add_writes_jsonl_without_profile_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                result = CliRunner().invoke(app, ["memory", "add", "最近写代码很烦，听到人声会更乱"])

                self.assertEqual(result.exit_code, 0, result.output)
                records = read_memory_logs()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["id"], "mem-000001")
                self.assertEqual(records[0]["sequence"], 1)
                self.assertIn("写代码", records[0]["body"])

                store = TonepathStore()
                try:
                    summary = store.profile_summary()
                finally:
                    store.close()
                self.assertEqual(summary["sessions"], 0)
                self.assertEqual(summary["feedback"], 0)
                self.assertEqual(summary["profile_rules"], 0)

    def test_memory_add_reads_multiline_stdin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                result = CliRunner().invoke(app, ["memory", "add", "--stdin"], input="第一行\n第二行\n")

                self.assertEqual(result.exit_code, 0, result.output)
                text = memory_log_path().read_text(encoding="utf-8")
                self.assertIn("第一行\\n第二行", text)

    def test_memory_add_assigns_unique_ids_with_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}, clear=True):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    list(executor.map(lambda index: add_memory_log(f"memory note {index}", source="test"), range(20)))

                records = read_memory_logs()
                sequences = [record["sequence"] for record in records]
                ids = [record["id"] for record in records]
                self.assertEqual(sequences, list(range(1, 21)))
                self.assertEqual(len(ids), len(set(ids)))

    def test_memory_show_displays_profile_guidance_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}, clear=True):
                result = CliRunner().invoke(app, ["memory", "show"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Tonepath Memory Profile", result.output)
                self.assertIn("No consolidated memory profile yet", result.output)

    def test_memory_edit_opens_profile_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "EDITOR": "true"}, clear=True):
                with patch("tonepath.cli.subprocess.run") as run:
                    result = CliRunner().invoke(app, ["memory", "edit"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue(memory_profile_path().exists())
                run.assert_called_once()
                self.assertEqual(Path(run.call_args.args[0][-1]), memory_profile_path())

    def test_memory_edit_reports_missing_editor_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "EDITOR": "/definitely/missing/editor"}, clear=True):
                result = CliRunner().invoke(app, ["memory", "edit"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("Editor failed", plain_output(result.output))

    def test_memory_consolidate_llm_requires_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                add_result = CliRunner().invoke(app, ["memory", "add", "最近写代码很烦，听到人声会更乱"])
                self.assertEqual(add_result.exit_code, 0, add_result.output)

                result = CliRunner().invoke(app, ["memory", "consolidate", "--llm"])

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("requires --confirm", plain_output(result.output))

    def test_memory_consolidate_llm_updates_profile_and_checkpoint(self) -> None:
        response = llm_response(
            {
                "profile_markdown": "# Tonepath Memory Profile\n\n## Listening Context\n\n写代码时偏好低人声、低刺激，但需要一点节奏。"
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                add_result = CliRunner().invoke(app, ["memory", "add", "最近写代码很烦，听到人声会更乱"])
                self.assertEqual(add_result.exit_code, 0, add_result.output)

                with patch("tonepath.memory.urllib.request.urlopen", return_value=response) as mocked:
                    result = CliRunner().invoke(app, ["memory", "consolidate", "--llm", "--confirm"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Memory profile updated", result.output)
                profile_text = memory_profile_path().read_text(encoding="utf-8")
                self.assertIn("低人声", profile_text)
                evidence_paths = list((home / "cache" / "memory").glob("*/evidence.json"))
                self.assertTrue(evidence_paths)
                evidence_text = evidence_paths[0].read_text(encoding="utf-8")
                self.assertNotIn(str(Path(tmp)), evidence_text)
                self.assertNotIn("secret", evidence_text)

                store = TonepathStore()
                try:
                    self.assertEqual(store.get_app_state("memory:last_consolidated_sequence"), "1")
                finally:
                    store.close()

                request = mocked.call_args.args[0]
                payload = json.loads(request.data.decode("utf-8"))
                system_prompt = payload["messages"][0]["content"]
                self.assertIn("listening context only", system_prompt)

    def test_memory_evidence_redacts_sensitive_text_without_changing_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                raw_path = "/Users/von/private/music-notes.md"
                raw_key = "sk-proj-abcdefghijklmnopqrstuvwxyz"
                profile_key = "Bearer abcdefghijklmnopqrstuvwxyz123456"
                add_memory_log(f"Use {raw_path} but never upload {raw_key}", source="test")
                memory_profile_path().parent.mkdir(parents=True, exist_ok=True)
                memory_profile_path().write_text(f"# Tonepath Memory Profile\n\nToken: {profile_key}\n", encoding="utf-8")

                store = TonepathStore()
                try:
                    evidence = build_memory_evidence(store)
                finally:
                    store.close()

                serialized = json.dumps(evidence, ensure_ascii=False)
                self.assertTrue(evidence["privacy"]["contains_api_keys"])
                self.assertTrue(evidence["privacy"]["contains_absolute_paths"])
                self.assertIn("[redacted-api-key]", serialized)
                self.assertIn("[redacted-absolute-path]", serialized)
                self.assertNotIn(raw_key, serialized)
                self.assertNotIn(raw_path, serialized)
                self.assertIn(raw_key, memory_log_path().read_text(encoding="utf-8"))
                self.assertIn(profile_key, memory_profile_path().read_text(encoding="utf-8"))

    def test_memory_consolidate_llm_wraps_unparseable_response(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"not-json"
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                with patch("tonepath.memory.urllib.request.urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "unparseable response"):
                        consolidate_memory_with_llm({"run_id": "r1", "new_memory_logs": []})

    def test_memory_consolidate_llm_requires_profile_markdown(self) -> None:
        response = llm_response({"wrong": "shape"})
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                with patch("tonepath.memory.urllib.request.urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, "profile_markdown"):
                        consolidate_memory_with_llm({"run_id": "r1", "new_memory_logs": []})

    def test_memory_suggest_llm_writes_pending_suggestions(self) -> None:
        response = llm_response(
            {
                "suggestions": [
                    {
                        "suggestion_id": "focus-low-vocal-low-stim",
                        "scope": "focus",
                        "rule_type": "prefer_lower_vocalness",
                        "target": "vocalness",
                        "threshold": 0.35,
                        "weight": 0.6,
                        "confidence": "medium",
                        "rationale": "写代码时减少人声和尖锐刺激。",
                        "evidence_count": 3,
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                memory_profile_path().parent.mkdir(parents=True)
                memory_profile_path().write_text("# Tonepath Memory Profile\n\n写代码时减少人声和尖锐刺激。\n", encoding="utf-8")
                add_result = CliRunner().invoke(app, ["memory", "add", "今天写代码时还是不想听人声"])
                self.assertEqual(add_result.exit_code, 0, add_result.output)

                with patch("tonepath.profile.urllib.request.urlopen", return_value=response) as mocked:
                    result = CliRunner().invoke(app, ["memory", "suggest", "--llm", "--confirm"])

                self.assertEqual(result.exit_code, 0, result.output)
                suggestion_paths = list((home / "cache" / "profile").glob("*/suggestions.json"))
                self.assertTrue(suggestion_paths)
                payload = json.loads(suggestion_paths[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["source"], "memory-llm")
                self.assertEqual(payload["suggestions"][0]["suggestion_id"], "focus-low-vocal-low-stim")
                self.assertEqual(payload["suggestions"][0]["source"], "memory-llm-deepseek")

                inspect = CliRunner().invoke(app, ["profile", "inspect", "--json"])
                self.assertEqual(inspect.exit_code, 0, inspect.output)
                inspect_payload = json.loads(inspect.output)
                self.assertEqual(inspect_payload["pending_suggestions"][0]["source"], "memory-llm-deepseek")
                self.assertIn("写代码", inspect_payload["pending_suggestions"][0]["rationale"])

                request = mocked.call_args.args[0]
                request_payload = json.loads(request.data.decode("utf-8"))
                user_message = json.loads(request_payload["messages"][1]["content"])
                self.assertIn("profile_memory_markdown", user_message)
                self.assertIn("Tonepath Memory Context", user_message["profile_memory_markdown"])

    def test_memory_suggest_invalid_suggestion_does_not_write_active_rule(self) -> None:
        response = llm_response(
            {
                "suggestions": [
                    {
                        "suggestion_id": "bad",
                        "scope": "focus",
                        "rule_type": "invent_new_rule",
                        "target": "vocalness",
                        "threshold": 0.3,
                        "weight": 0.5,
                        "confidence": "medium",
                        "rationale": "bad",
                        "evidence_count": 1,
                    }
                ]
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home"), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                add_result = CliRunner().invoke(app, ["memory", "add", "不要人声"])
                self.assertEqual(add_result.exit_code, 0, add_result.output)
                with patch("tonepath.profile.urllib.request.urlopen", return_value=response):
                    result = CliRunner().invoke(app, ["memory", "suggest", "--llm", "--confirm"])

                self.assertNotEqual(result.exit_code, 0)
                store = TonepathStore()
                try:
                    self.assertEqual(store.profile_summary()["profile_rules"], 0)
                finally:
                    store.close()

    def test_memory_log_ignores_records_with_invalid_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}, clear=True):
                memory_log_path().parent.mkdir(parents=True, exist_ok=True)
                memory_log_path().write_text(
                    '{"body":"bad","created_at":"now","id":"mem-bad","sequence":"oops","source":"test"}\n'
                    '{"body":"good","created_at":"now","id":"mem-000002","sequence":2,"source":"test"}\n',
                    encoding="utf-8",
                )

                records = read_memory_logs()
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["id"], "mem-000002")
                self.assertEqual(records[0]["sequence"], 2)

    def test_memory_artifacts_are_owner_only(self) -> None:
        response = llm_response({"profile_markdown": "# Tonepath Memory Profile\n\nPrivate profile.\n"})
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": "secret"}, clear=True):
                add_result = CliRunner().invoke(app, ["memory", "add", "private memory"])
                self.assertEqual(add_result.exit_code, 0, add_result.output)
                with patch("tonepath.memory.urllib.request.urlopen", return_value=response):
                    result = CliRunner().invoke(app, ["memory", "consolidate", "--llm", "--confirm"])
                self.assertEqual(result.exit_code, 0, result.output)
                evidence_path = next((home / "cache" / "memory").glob("*/evidence.json"))

                self.assertEqual(stat.S_IMODE(memory_log_path().stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(memory_profile_path().stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(evidence_path.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(memory_log_path().parent.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(evidence_path.parent.stat().st_mode), 0o700)

    def test_profile_delete_all_preserves_raw_memory_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                add_result = CliRunner().invoke(app, ["memory", "add", "我想保留这段树洞"])
                self.assertEqual(add_result.exit_code, 0, add_result.output)
                memory_profile_path().parent.mkdir(parents=True, exist_ok=True)
                memory_profile_path().write_text("# Tonepath Memory Profile\n\n用户可编辑内容。\n", encoding="utf-8")

                result = CliRunner().invoke(app, ["profile", "delete", "--all"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue(memory_log_path().exists())
                self.assertTrue(memory_profile_path().exists())
                self.assertIn("我想保留这段树洞", memory_log_path().read_text(encoding="utf-8"))


def llm_response(content: dict[str, object]) -> MagicMock:
    response = MagicMock()
    body = {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}
    response.__enter__.return_value.read.return_value = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return response
