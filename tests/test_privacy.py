import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.memory import LAST_CONSOLIDATED_SEQUENCE_KEY
from tonepath.models import ProfileRule, SessionPlan, SessionRequest, Track, TrackFeatures
from tonepath.planner import build_phases
from tonepath.privacy import (
    ALL_PERSONAL_CATEGORIES,
    build_privacy_inventory,
    delete_profile,
    execute_privacy_delete,
    export_personal_data,
    plan_privacy_delete,
    privacy_status,
)


class PrivacyInventoryTest(unittest.TestCase):
    def test_missing_home_inventory_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "missing-home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                inventory = build_privacy_inventory()

            self.assertFalse(home.exists())
            self.assertFalse(inventory.database_exists)
            self.assertEqual(
                [category.id for category in inventory.categories],
                ["memory", "personalization", "history", "library-evidence", "models-storage"],
            )
            self.assertTrue(all(category.record_count == 0 for category in inventory.categories))

    def test_inventory_counts_known_data_without_following_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            external = Path(tmp) / "external"
            external.mkdir()
            (external / "large.bin").write_bytes(b"x" * 4096)
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                memory_dir = home / "memory"
                memory_dir.mkdir(parents=True)
                (memory_dir / "profile.md").write_text("private", encoding="utf-8")
                (memory_dir / "outside").symlink_to(external, target_is_directory=True)
                (home / "cache" / "models").mkdir(parents=True)
                (home / "cache" / "models" / "model.bin").write_bytes(b"model")

                inventory = build_privacy_inventory()
                store.close()

            categories = {category.id: category for category in inventory.categories}
            self.assertEqual(categories["memory"].file_count, 1)
            self.assertEqual(categories["memory"].file_size_bytes, len(b"private"))
            self.assertEqual(categories["personalization"].records["feedback"], 1)
            self.assertEqual(categories["personalization"].records["profile_rules"], 1)
            self.assertEqual(categories["history"].records["sessions"], 1)
            self.assertEqual(categories["library-evidence"].records["tracks"], 1)
            self.assertEqual(categories["models-storage"].file_size_bytes, len(b"model"))

    def test_external_processing_reports_policy_without_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            (home / "config.toml").write_text(
                "network_mode = \"online-opt-in\"\n"
                "[privacy]\n"
                "send_to_llm = true\n",
                encoding="utf-8",
            )
            secret = "sk-secret-must-not-appear"
            with patch.dict(
                os.environ,
                {
                    "TONEPATH_HOME": str(home),
                    "TONEPATH_LLM_PROVIDER": "deepseek",
                    "DEEPSEEK_API_KEY": secret,
                },
                clear=True,
            ):
                payload = build_privacy_inventory().to_payload()

            rendered = json.dumps(payload)
            self.assertEqual(payload["external_processing"]["provider"], "deepseek")
            self.assertTrue(payload["external_processing"]["key_present"])
            self.assertEqual(payload["external_processing"]["transmission_history"], "not recorded")
            self.assertNotIn(secret, rendered)


class PrivacyExportTest(unittest.TestCase):
    def test_export_is_private_and_omits_paths_secrets_database_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            music = root / "private-music"
            music.mkdir()
            audio = music / "secret-song.mp3"
            audio.write_bytes(b"audio")
            secret = "sk-export-secret"
            with patch.dict(
                os.environ,
                {"TONEPATH_HOME": str(home), "DEEPSEEK_API_KEY": secret},
                clear=True,
            ):
                store = populated_store(home, track_path=audio)
                (home / "memory" / "logs").mkdir(parents=True)
                (home / "memory" / "profile.md").write_text(
                    f"My path is {music} and token is {secret}.",
                    encoding="utf-8",
                )
                (home / "memory" / "logs" / "memory-log.jsonl").write_text(
                    json.dumps(
                        {
                            "id": "mem-000001",
                            "sequence": 1,
                            "created_at": "2026-01-01T00:00:00Z",
                            "source": "cli",
                            "body": f"Use {audio} with Bearer {secret}",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (home / "config.toml").write_text(
                    f"music_dirs = [\"{music}\"]\n"
                    f"data_dir = \"{home}\"\n"
                    "network_mode = \"offline\"\n",
                    encoding="utf-8",
                )
                output = root / "export"

                export_personal_data(output, store=store)
                store.close()

            files = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(
                files,
                {
                    "README.md",
                    "history.json",
                    "manifest.json",
                    "memory/memory-log.jsonl",
                    "memory/profile.md",
                    "personalization.json",
                    "settings.json",
                },
            )
            exported = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
            self.assertNotIn(secret, exported)
            self.assertNotIn(str(root), exported)
            self.assertNotIn("tonepath.db", exported)
            self.assertNotIn("secret-song.mp3", exported)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            for path in output.rglob("*"):
                expected = 0o700 if path.is_dir() else 0o600
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected)


class PrivacyDeleteTest(unittest.TestCase):
    def test_preview_is_zero_write_and_warns_when_memory_rules_remain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                make_personal_files(home)
                before = store.profile_summary()

                plan = plan_privacy_delete(("memory",), store=store)

                self.assertEqual(store.profile_summary(), before)
                self.assertTrue((home / "memory" / "profile.md").exists())
                self.assertTrue(any("active rules" in warning.lower() for warning in plan.warnings))
                store.close()

    def test_delete_memory_removes_memory_only_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                make_personal_files(home)
                plan = plan_privacy_delete(("memory",), store=store)

                result = execute_privacy_delete(plan, store=store)

                self.assertFalse((home / "memory").exists())
                self.assertFalse((home / "cache" / "memory").exists())
                self.assertIsNone(store.get_app_state(LAST_CONSOLIDATED_SEQUENCE_KEY))
                self.assertEqual(store.profile_summary()["profile_rules"], 1)
                self.assertEqual(store.profile_summary()["feedback"], 1)
                self.assertFalse(result.failed)
                store.close()

    def test_delete_personalization_preserves_memory_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                make_personal_files(home)
                plan = plan_privacy_delete(("personalization",), store=store)

                execute_privacy_delete(plan, store=store)

                summary = store.profile_summary()
                self.assertEqual(summary["feedback"], 0)
                self.assertEqual(summary["profile_rules"], 0)
                self.assertEqual(summary["sessions"], 1)
                self.assertTrue((home / "memory" / "profile.md").exists())
                self.assertFalse((home / "profile").exists())
                self.assertFalse((home / "cache" / "profile").exists())
                store.close()

    def test_delete_history_preserves_feedback_but_clears_session_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                make_personal_files(home)
                plan = plan_privacy_delete(("history",), store=store)

                execute_privacy_delete(plan, store=store)

                summary = store.profile_summary()
                self.assertEqual(summary["sessions"], 0)
                self.assertEqual(summary["session_phases"], 0)
                self.assertEqual(summary["plays"], 0)
                self.assertEqual(summary["feedback"], 1)
                row = store.conn.execute("SELECT session_id FROM feedback").fetchone()
                self.assertIsNone(row["session_id"])
                self.assertFalse((home / "cache" / "audit").exists())
                self.assertTrue((home / "memory" / "profile.md").exists())
                store.close()

    def test_all_personal_preserves_library_models_config_and_music(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            music = root / "music"
            music.mkdir()
            audio = music / "song.mp3"
            audio.write_bytes(b"audio")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home, track_path=audio)
                make_personal_files(home)
                (home / "config.toml").write_text("network_mode = \"offline\"\n", encoding="utf-8")
                (home / "cache" / "models").mkdir(parents=True)
                (home / "cache" / "models" / "model.bin").write_bytes(b"model")
                plan = plan_privacy_delete(ALL_PERSONAL_CATEGORIES, store=store)

                execute_privacy_delete(plan, store=store)

                summary = store.profile_summary()
                self.assertEqual(summary["tracks"], 1)
                self.assertEqual(summary["track_features"], 1)
                self.assertEqual(summary["sessions"], 0)
                self.assertEqual(summary["feedback"], 0)
                self.assertEqual(summary["profile_rules"], 0)
                self.assertTrue((home / "config.toml").exists())
                self.assertTrue((home / "cache" / "models" / "model.bin").exists())
                self.assertTrue(audio.exists())
                store.close()

    def test_changed_data_invalidates_delete_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                plan = plan_privacy_delete(("personalization",), store=store)
                store.record_feedback("like")

                with self.assertRaisesRegex(RuntimeError, "changed since the preview"):
                    execute_privacy_delete(plan, store=store)
                store.close()

    def test_read_only_category_cannot_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                with self.assertRaisesRegex(ValueError, "read-only"):
                    plan_privacy_delete(("library-evidence",))

    def test_repeated_delete_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                make_personal_files(home)
                first = execute_privacy_delete(
                    plan_privacy_delete(("memory",), store=store),
                    store=store,
                )
                second = execute_privacy_delete(
                    plan_privacy_delete(("memory",), store=store),
                    store=store,
                )

                self.assertIn("memory", first.changed_categories)
                self.assertFalse(second.failed)
                self.assertTrue(all(item.status == "already_absent" for item in second.items))
                store.close()

    def test_partial_filesystem_failure_is_reported_without_hiding_db_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                make_personal_files(home)
                plan = plan_privacy_delete(("personalization",), store=store)
                real_rmtree = shutil.rmtree

                def fail_profile(path: Path) -> None:
                    if Path(path) == home / "profile":
                        raise PermissionError("profile directory is locked")
                    real_rmtree(path)

                with patch("tonepath.privacy.shutil.rmtree", side_effect=fail_profile):
                    result = execute_privacy_delete(plan, store=store)

                self.assertTrue(result.failed)
                self.assertEqual(store.profile_summary()["feedback"], 0)
                failed = [item for item in result.items if item.status == "failed"]
                self.assertEqual(len(failed), 1)
                self.assertIn("locked", failed[0].message)
                self.assertTrue((home / "profile").exists())
                self.assertFalse((home / "cache" / "profile").exists())
                store.close()

    def test_delete_profile_keeps_legacy_all_profile_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = populated_store(Path(tmp))
            delete_profile(store)
            summary = store.profile_summary()
            self.assertEqual(summary["sessions"], 0)
            self.assertEqual(summary["feedback"], 0)
            self.assertEqual(summary["profile_rules"], 0)
            self.assertEqual(summary["tracks"], 1)
            store.close()


class PrivacyCliTest(unittest.TestCase):
    def test_inspect_json_is_read_only_and_has_stable_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "missing"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                result = CliRunner().invoke(app, ["privacy", "inspect", "--json"])

            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.output)
            self.assertEqual(payload["schema"], "tonepath-privacy-inventory-v1")
            self.assertFalse(home.exists())

    def test_delete_without_confirm_only_prints_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                store.close()
                result = CliRunner().invoke(
                    app,
                    ["privacy", "delete", "--category", "personalization"],
                )
                check = TonepathStore(home / "tonepath.db")

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Preview only", result.output)
            self.assertEqual(check.profile_summary()["feedback"], 1)
            check.close()

    def test_delete_with_confirm_executes_selected_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = populated_store(home)
                store.close()

                result = CliRunner().invoke(
                    app,
                    ["privacy", "delete", "--category", "personalization", "--confirm"],
                )
                check = TonepathStore(home / "tonepath.db")

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Privacy deletion result", result.output)
            self.assertEqual(check.profile_summary()["feedback"], 0)
            self.assertEqual(check.profile_summary()["sessions"], 1)
            check.close()


def populated_store(home: Path, track_path: Path | None = None) -> TonepathStore:
    home.mkdir(parents=True, exist_ok=True)
    store = TonepathStore(home / "tonepath.db")
    path = track_path or home / "song.mp3"
    if not path.exists():
        path.write_bytes(b"audio")
    track_id = store.upsert_track(
        Track(
            id=None,
            path=path,
            file_hash="hash",
            mtime=path.stat().st_mtime,
            title="Song",
            artist="Artist",
            album=None,
            genre=None,
            duration=120.0,
            format="mp3",
        )
    )
    store.upsert_features(
        TrackFeatures(
            track_id=track_id,
            bpm=90.0,
            loudness=-18.0,
            energy=0.3,
            vocalness=0.1,
            arousal_estimate=0.3,
            valence_estimate=0.6,
            feature_source="test",
            confidence="high",
        )
    )
    request = SessionRequest("focus", "irritated", "focus", 1800)
    session_id = store.save_session(SessionPlan(request, tuple(build_phases(request))))
    store.start_play(session_id, track_id)
    store.record_feedback("like", session_id=session_id, track_id=track_id)
    store.upsert_profile_rule(ProfileRule(None, "focus:prefer_lower_vocalness:vocalness", "0.3", "test", "high"))
    store.set_app_state(LAST_CONSOLIDATED_SEQUENCE_KEY, "1")
    return store


def make_personal_files(home: Path) -> None:
    for relative, content in (
        ("memory/profile.md", "# Memory"),
        ("memory/logs/memory-log.jsonl", "{}\n"),
        ("cache/memory/run/evidence.json", "{}"),
        ("profile/memory.md", "# Profile"),
        ("cache/profile/run/suggestions.json", "{}"),
        ("cache/audit/run/evidence.json", "{}"),
    ):
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
