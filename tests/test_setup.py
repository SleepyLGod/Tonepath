import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tonepath import config
from tonepath.setup import SetupDraft, setup_review, validate_music_directories


class SetupDomainTest(unittest.TestCase):
    def test_from_config_expands_music_directories_for_consistent_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HOME": tmp}):
            settings = replace(config.default_config(), music_dirs=("~/Music",))

            draft = SetupDraft.from_config(settings)

            self.assertEqual(draft.music_dirs, (str(Path(tmp) / "Music"),))
            self.assertEqual(draft.remove_music_dir(Path("~/Music")).music_dirs, ())

    def test_replace_music_directories_rejects_blank_entries(self) -> None:
        draft = SetupDraft.from_config(config.default_config())

        for raw_path in ("", "   "):
            with self.subTest(raw_path=raw_path), self.assertRaisesRegex(ValueError, "cannot be empty"):
                draft.replace_music_dirs((raw_path,))

    def test_replace_music_directories_expands_and_deduplicates_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"HOME": tmp}):
            draft = SetupDraft.from_config(config.default_config())

            updated = draft.replace_music_dirs(("~/Music", "/tmp/second", "~/Music"))

            self.assertEqual(updated.music_dirs, (str(Path(tmp) / "Music"), "/tmp/second"))

    def test_validate_music_directories_rejects_blank_entries(self) -> None:
        for raw_path in ("", "   "):
            with self.subTest(raw_path=raw_path), self.assertRaisesRegex(ValueError, "cannot be empty"):
                validate_music_directories((raw_path,))

    def test_music_directory_changes_preserve_unrelated_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            first.mkdir()
            second.mkdir()
            draft = SetupDraft.from_config(config.default_config()).replace_music_dirs((str(first),))

            updated = draft.add_music_dir(second)

            self.assertEqual(updated.music_dirs, (str(first), str(second)))
            self.assertEqual(updated.remove_music_dir(first).music_dirs, (str(second),))

    def test_private_experience_disables_external_text_processing(self) -> None:
        draft = SetupDraft.from_config(config.default_config()).with_experience(
            "private",
            send_to_llm=True,
            provider="qwen",
        )

        settings = draft.to_config(config.default_config())

        self.assertEqual(settings.experience.mode, "private")
        self.assertEqual(settings.network_mode, "offline")
        self.assertFalse(settings.models.allow_online)
        self.assertFalse(settings.privacy.send_to_llm)
        self.assertEqual(settings.llm.provider, "qwen")

    def test_smart_experience_keeps_consent_explicit(self) -> None:
        draft = SetupDraft.from_config(config.default_config()).with_experience(
            "smart",
            send_to_llm=False,
            provider="qwen",
        )

        settings = draft.to_config(config.default_config())

        self.assertEqual(settings.experience.mode, "smart")
        self.assertEqual(settings.network_mode, "online-opt-in")
        self.assertTrue(settings.models.allow_online)
        self.assertFalse(settings.privacy.send_to_llm)
        self.assertEqual(settings.llm.provider, "qwen")
        self.assertEqual(settings.models.mode, "full")

    def test_custom_experience_preserves_existing_network_policy(self) -> None:
        defaults = config.default_config()
        current = replace(
            defaults,
            network_mode="offline",
            models=replace(defaults.models, allow_online=False),
            experience=config.ExperienceConfig(mode="custom"),
        )
        draft = SetupDraft.from_config(current).with_experience(
            "custom",
            send_to_llm=True,
            provider="qwen",
        )

        settings = draft.to_config(current)

        self.assertEqual(settings.network_mode, "offline")
        self.assertFalse(settings.models.allow_online)

    def test_invalid_music_directory_is_rejected_without_writing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            missing = Path(tmp) / "missing"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    validate_music_directories((str(missing),))

                self.assertFalse(config.config_path().exists())

    def test_review_explains_local_data_ai_consent_and_model_state(self) -> None:
        draft = SetupDraft.from_config(config.default_config()).with_experience(
            "smart",
            send_to_llm=True,
            provider="deepseek",
        )

        review = setup_review(draft, model_ready=False, provider_key_ready=False)

        self.assertIn("Music stays local", review)
        self.assertIn("DeepSeek key missing", review)
        self.assertIn("Local models: available to set up", review)
        self.assertNotIn("API_KEY", review)


if __name__ == "__main__":
    unittest.main()
