import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Input, TextArea
from textual.worker import WorkerState

from tonepath import config
from tonepath.preparation import PreparationResult, ScanSummary
from tonepath.planner import plan_session
from tonepath.readiness import LibraryStatus
from tonepath.setup import SetupDraft
from tonepath.tui import TonepathApp
from tonepath.tui_privacy import deletion_task_blocker
from tonepath.tui_setup import SetupOutcome, SetupScreen


class TuiSetupTest(unittest.IsolatedAsyncioTestCase):
    async def wait_until(self, pilot: object, predicate: object, attempts: int = 100) -> None:
        for _ in range(attempts):
            if predicate():
                return
            await pilot.pause(0.05)
        self.fail("Timed out waiting for TUI setup state.")

    async def test_missing_config_auto_opens_getting_started(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    screen = app.screen
                    self.assertTrue(screen.first_run)
                    self.assertIn("Getting Started", screen.query_one("#setup-heading").render().plain)
                    self.assertFalse(config.config_path().exists())
                    await pilot.press("escape")
                    await self.wait_until(pilot, lambda: not isinstance(app.screen, SetupScreen))
                    self.assertFalse(config.config_path().exists())

    async def test_existing_local_state_without_config_is_not_treated_as_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            (home / "legacy-state.txt").write_text("existing", encoding="utf-8")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await pilot.pause()
                    self.assertNotIsInstance(app.screen, SetupScreen)
                    self.assertFalse(config.config_path().exists())

    async def test_c_opens_setup_only_outside_request_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.preset_config("private", music_dir=music))
                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt = app.query_one("#prompt-input", Input)
                    prompt.focus()
                    await pilot.press("c")
                    self.assertEqual(prompt.value, "c")
                    self.assertNotIsInstance(app.screen, SetupScreen)

                    await pilot.press("ctrl+o")
                    memory = app.query_one("#memory-input", TextArea)
                    await pilot.press("c")
                    self.assertEqual(memory.text, "c")
                    self.assertNotIsInstance(app.screen, SetupScreen)
                    await pilot.press("ctrl+o")
                    memory.blur()
                    prompt.blur()
                    await pilot.press("c")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    self.assertFalse(app.screen.first_run)
                    self.assertIn("Current Setup", app.screen.query_one("#setup-options").border_title)

    async def test_reconfiguration_changes_one_area_and_preserves_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                original = config.preset_config("private", music_dir=music)
                config.write_config(original)
                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.query_one("#prompt-input", Input).blur()
                    await pilot.press("c")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    screen = app.screen

                    for _ in range(3):
                        await pilot.press("down")
                    await pilot.press("enter")
                    self.assertEqual(screen.state, "local-data")
                    await pilot.press("down")
                    await pilot.press("enter")
                    self.assertEqual(screen.state, "summary")

                    for _ in range(5):
                        await pilot.press("down")
                    await pilot.press("enter")
                    await pilot.press("enter")
                    await pilot.press("enter")
                    await self.wait_until(pilot, lambda: not isinstance(app.screen, SetupScreen))

                    updated = config.load_config()
                    self.assertEqual(updated.music_dirs, original.music_dirs)
                    self.assertEqual(updated.experience, original.experience)
                    self.assertEqual(updated.models, original.models)
                    self.assertFalse(updated.privacy.store_play_history)

    async def test_add_music_directory_starts_with_an_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.preset_config("private", music_dir=music))
                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.query_one("#prompt-input", Input).blur()
                    await pilot.press("c")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    screen = app.screen
                    await pilot.press("enter")
                    self.assertEqual(screen.state, "music-menu")
                    await pilot.press("down")
                    await pilot.press("enter")
                    self.assertEqual(screen.state, "music-input")
                    self.assertEqual(screen.query_one("#setup-music-input", Input).value, "")

    async def test_setup_music_input_owns_arrows_without_seeking_player(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.preset_config("private", music_dir=music))
                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.query_one("#prompt-input", Input).blur()
                    await pilot.press("c")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    screen = app.screen
                    with patch.object(app, "seek_playback") as seek:
                        await pilot.press("left")
                        await pilot.press("right")
                        seek.assert_not_called()

                        await pilot.press("enter")
                        await pilot.press("down")
                        await pilot.press("enter")
                        self.assertEqual(screen.state, "music-input")
                        music_input = screen.query_one("#setup-music-input", Input)
                        music_input.value = "abcd"
                        music_input.cursor_position = 2

                        await pilot.press("left")
                        self.assertEqual(music_input.cursor_position, 1)
                        await pilot.press("right")
                        self.assertEqual(music_input.cursor_position, 2)
                        await pilot.press("up")
                        await pilot.press("down")
                        seek.assert_not_called()

    async def test_finish_setup_keeps_invalid_music_error_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            missing = Path(tmp) / "missing"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    screen = app.screen
                    screen.draft = screen.draft.replace_music_dirs((str(missing),))

                    screen.finish_setup(prepare=False, setup_models=False)
                    await pilot.pause()

                    self.assertEqual(screen.state, "music-input")
                    self.assertIn("does not exist", screen.status_message)
                    self.assertIn("does not exist", screen.query_one("#setup-status").render().plain)

    async def test_first_run_private_flow_saves_only_after_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await self.wait_until(pilot, lambda: isinstance(app.screen, SetupScreen))
                    screen = app.screen
                    music_input = screen.query_one("#setup-music-input", Input)
                    music_input.value = str(music)
                    await pilot.press("enter")
                    self.assertEqual(screen.state, "experience")
                    self.assertFalse(config.config_path().exists())

                    await pilot.press("enter")
                    self.assertEqual(screen.state, "review")
                    await pilot.press("enter")
                    self.assertEqual(screen.state, "prepare-choice")
                    await pilot.press("enter")
                    await self.wait_until(pilot, lambda: not isinstance(app.screen, SetupScreen))

                    settings = config.load_config()
                    self.assertEqual(settings.music_dirs, (str(music),))
                    self.assertEqual(settings.experience.mode, "private")
                    self.assertFalse(app.setup_prepare_busy)

    async def test_background_prepare_keeps_queue_and_player_controls_responsive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            started = threading.Event()
            release = threading.Event()
            status = LibraryStatus(1, 1, 0, 1, 1, 1, 1)

            def slow_prepare(*args: object, **kwargs: object) -> PreparationResult:
                started.set()
                release.wait(timeout=5)
                return PreparationResult(
                    scan=ScanSummary(1, 1, 0, 0),
                    failures=(),
                    status=status,
                    runtime_ready=True,
                    affect_ready=True,
                )

            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True), patch(
                "tonepath.tui.tonepath_preparation.run_preparation",
                side_effect=slow_prepare,
            ):
                settings = config.preset_config("private", music_dir=music)
                config.write_config(settings)
                app = TonepathApp()
                async with app.run_test() as pilot:
                    plan = plan_session("我想安静工作二十分钟")
                    runner = SimpleNamespace(
                        queue=["current"],
                        active_plan=lambda: plan,
                        current=lambda: None,
                        upcoming=lambda: (),
                    )
                    app.runner = runner
                    original_mode = app.playback_mode
                    app.on_setup_complete(
                        SetupOutcome(
                            settings=settings,
                            prepare_requested=True,
                            setup_models=False,
                        )
                    )
                    await self.wait_until(pilot, started.is_set)

                    self.assertTrue(app.setup_prepare_busy)
                    app.query_one("#prompt-input", Input).blur()
                    await pilot.press("m")
                    self.assertNotEqual(app.playback_mode, original_mode)
                    self.assertIs(app.runner, runner)
                    self.assertEqual(app.runner.queue, ["current"])
                    self.assertIn("preparation is still running", deletion_task_blocker(app, ("history",)) or "")

                    release.set()
                    await self.wait_until(pilot, lambda: not app.setup_prepare_busy)
                    self.assertIs(app.runner, runner)

    async def test_background_prepare_restores_ready_status_after_finding_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            status = LibraryStatus(1, 1, 0, 1, 0, 0, 0)
            result = PreparationResult(
                scan=ScanSummary(1, 1, 0, 0),
                failures=(),
                status=status,
                runtime_ready=False,
                affect_ready=False,
            )
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.preset_config("private", music_dir=music))
                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.playback_status = "No tracks"
                    worker = SimpleNamespace(result=result)
                    with patch.object(app, "library_count", return_value=1):
                        app.finish_setup_preparation_worker(worker, WorkerState.SUCCESS)  # type: ignore[arg-type]
                    await pilot.pause()

                    self.assertEqual(app.playback_status, "Ready")
                    self.assertIn("Library preparation finished", app.setup_prepare_status)

    async def test_help_always_lists_setup_and_command_bar_only_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            music = Path(tmp) / "music"
            music.mkdir()
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.preset_config("private", music_dir=music))
                app = TonepathApp()
                async with app.run_test() as pilot:
                    self.assertIn("c          Setup", app.help_panel_text())
                    self.assertIn("c  Setup", app.command_bar_renderable().plain)
                    app.readiness = "Model setup available"
                    self.assertNotIn("c  Setup", app.command_bar_renderable().plain)
                    app.readiness = "Ready for TUI"
                    self.assertNotIn("c  Setup", app.command_bar_renderable().plain)


class SetupOutcomeTest(unittest.TestCase):
    def test_summary_uses_normal_directory_pluralization(self) -> None:
        defaults = config.default_config()
        one = SetupScreen(replace(defaults, music_dirs=("/one",)), first_run=False, model_ready=False)
        two = SetupScreen(replace(defaults, music_dirs=("/one", "/two")), first_run=False, model_ready=False)

        one_music = next(row for row in one.state_options()[0] if row[0] == "music")
        two_music = next(row for row in two.state_options()[0] if row[0] == "music")

        self.assertEqual(one_music[2], "1 local directory")
        self.assertEqual(two_music[2], "2 local directories")

    def test_outcome_keeps_setup_choices_explicit(self) -> None:
        settings = SetupDraft.from_config(config.default_config()).to_config(config.default_config())
        outcome = SetupOutcome(settings=settings, prepare_requested=False, setup_models=False)

        self.assertFalse(outcome.prepare_requested)
        self.assertFalse(outcome.setup_models)


if __name__ == "__main__":
    unittest.main()
