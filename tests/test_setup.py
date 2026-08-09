import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath import config
from tonepath.setup import SetupDraft, setup_review, validate_music_directories


class SetupDomainTest(unittest.TestCase):
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
        self.assertFalse(settings.privacy.send_to_llm)
        self.assertEqual(settings.llm.provider, "qwen")
        self.assertEqual(settings.models.mode, "full")

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
