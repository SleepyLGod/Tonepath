import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tonepath import config
from tonepath.cli import app
from tonepath.llm import active_provider, llm_doctor, parse_prompt_with_llm
from tonepath.model_runtime import clap_runtime_status, model_runtime_status, setup_clap_runtime, setup_essentia_tf_runtime


class RuntimeAndLlmTest(unittest.TestCase):
    def test_local_env_loader_does_not_override_process_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("TONEPATH_LLM_PROVIDER=qwen\nDEEPSEEK_API_KEY=from-file\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "from-process"}, clear=True):
                config.load_local_env(env_path)
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-process")
                self.assertEqual(os.environ["TONEPATH_LLM_PROVIDER"], "qwen")

    def test_model_doctor_reports_missing_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": tmp}, clear=True):
                result = CliRunner().invoke(app, ["models", "doctor"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Ready: False", result.output)
                self.assertIn("Run: uv run tonepath models setup essentia-tf", result.output)

    def test_model_setup_uses_workspace_local_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": tmp}, clear=True), patch(
                "tonepath.model_runtime.ensure_isolated_python311", return_value=Path("/isolated/python3.11")
            ), patch("tonepath.model_runtime.subprocess.run") as run, patch(
                "tonepath.model_runtime.download_essentia_models"
            ):
                setup_essentia_tf_runtime()

                status = model_runtime_status()
                self.assertTrue(str(status.runtime_dir).startswith(tmp))
                self.assertTrue(status.runner.exists())
                self.assertTrue(run.called)

    def test_clap_model_setup_uses_workspace_local_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": tmp}, clear=True), patch(
                "tonepath.model_runtime.ensure_isolated_python311", return_value=Path("/isolated/python3.11")
            ), patch("tonepath.model_runtime.subprocess.run") as run, patch("tonepath.model_runtime.download_clap_checkpoint"):
                setup_clap_runtime()

                status = clap_runtime_status()
                self.assertTrue(str(status.runtime_dir).startswith(tmp))
                self.assertTrue(status.runner.exists())
                self.assertTrue(run.called)

    def test_llm_doctor_is_redacted(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret", "TONEPATH_LLM_PROVIDER": "deepseek"}, clear=True):
            report = llm_doctor()

        self.assertIn("DEEPSEEK_API_KEY (configured)", report)
        self.assertIn("Secrets: not displayed", report)
        self.assertNotIn("secret", report)

    def test_active_provider_uses_config_with_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(
                    config.TonepathConfig(
                        music_dirs=("~/Music",),
                        data_dir=str(home),
                        player="mpv",
                        network_mode="offline",
                        privacy=config.PrivacyConfig(),
                        models=config.ModelConfig(),
                        experience=config.ExperienceConfig(),
                        llm=config.LlmConfig(provider="qwen"),
                    )
                )
                self.assertEqual(active_provider(), "qwen")

                with patch.dict(os.environ, {"TONEPATH_LLM_PROVIDER": "deepseek"}):
                    self.assertEqual(active_provider(), "deepseek")

    def test_llm_parse_sends_only_prompt_and_returns_structured_intent(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "source_state": "irritated",
                                "target_state": "focus",
                                "duration_min": 30,
                                "constraints": ["avoid_vocals"],
                            }
                        )
                    }
                }
            ]
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(body).encode("utf-8")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "secret", "TONEPATH_LLM_PROVIDER": "deepseek"}, clear=True), patch(
            "tonepath.llm.urllib.request.urlopen", return_value=response
        ) as urlopen:
            parsed = parse_prompt_with_llm("我现在很烦，想半小时后进入写代码状态，不要人声")

        self.assertEqual(parsed["target_state"], "focus")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIn("我现在很烦", payload["messages"][1]["content"])
        self.assertNotIn("Music directories", json.dumps(payload))

    def test_parse_llm_command_requires_key_without_printing_secret(self) -> None:
        with patch.dict(os.environ, {"TONEPATH_LLM_PROVIDER": "deepseek"}, clear=True):
            result = CliRunner().invoke(app, ["parse", "--llm", "我现在很烦"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("DEEPSEEK_API_KEY", result.output)


if __name__ == "__main__":
    unittest.main()
