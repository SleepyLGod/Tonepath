import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from textual.widgets import Input, TextArea

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import SessionPlan, SessionRequest
from tonepath.planner import build_phases, plan_session
from tonepath.privacy import (
    PrivacyDeleteItemResult,
    PrivacyDeleteResult,
)
from tonepath.tui import TonepathApp
from tonepath.tui_privacy import (
    PrivacyDataDeleted,
    PrivacyDeleteModal,
    PrivacyScreen,
    category_delete_completed,
    deletion_task_blocker,
)


class PrivacyScreenTest(unittest.IsolatedAsyncioTestCase):
    async def wait_until(self, pilot: object, predicate: object, attempts: int = 100) -> None:
        for _ in range(attempts):
            if predicate():
                return
            await pilot.pause(0.05)
        self.fail("Timed out waiting for TUI privacy state.")

    async def open_privacy(self, app: TonepathApp, pilot: object) -> PrivacyScreen:
        prompt = app.query_one("#prompt-input", Input)
        prompt.blur()
        await pilot.press("d")
        await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyScreen))
        screen = app.screen
        self.assertIsInstance(screen, PrivacyScreen)
        await self.wait_until(pilot, lambda: screen.inventory is not None and not screen.busy)
        return screen

    async def confirm_delete(self, app: TonepathApp, pilot: object) -> PrivacyScreen:
        await pilot.press("d")
        await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyDeleteModal))
        modal = app.screen
        self.assertIsInstance(modal, PrivacyDeleteModal)
        confirmation = modal.query_one("#privacy-confirm-input", Input)
        confirmation.value = "delete"
        await pilot.press("enter")
        await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyScreen))
        screen = app.screen
        self.assertIsInstance(screen, PrivacyScreen)
        await self.wait_until(pilot, lambda: not screen.busy)
        return screen

    async def test_d_opens_privacy_only_outside_text_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.default_config())
                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt = app.query_one("#prompt-input", Input)
                    prompt.focus()
                    await pilot.press("d")
                    self.assertEqual(prompt.value, "d")
                    self.assertNotIsInstance(app.screen, PrivacyScreen)

                    await pilot.press("ctrl+o")
                    memory = app.query_one("#memory-input", TextArea)
                    self.assertTrue(memory.has_focus)
                    await pilot.press("d")
                    self.assertEqual(memory.text, "d")
                    self.assertNotIsInstance(app.screen, PrivacyScreen)

                    await pilot.press("ctrl+o")
                    screen = await self.open_privacy(app, pilot)
                    self.assertEqual(len(screen.inventory.categories), 5)
                    self.assertIn("Data & Privacy", screen.query_one("#privacy-heading").render().plain)
                    self.assertIn("d          Data & Privacy", app.help_panel_text())
                    self.assertIn("d  Data", app.command_bar_renderable().plain)
                    await pilot.press("escape")

    async def test_browsing_preserves_current_runner_queue_and_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.default_config())
                app = TonepathApp()
                async with app.run_test() as pilot:
                    runner = SimpleNamespace(queue=["first", "second"])
                    app.runner = runner
                    app.playback = MagicMock()
                    app.playback_status = "Playing"
                    screen = await self.open_privacy(app, pilot)
                    first = screen.selected_category_id

                    await pilot.press("down")

                    self.assertNotEqual(screen.selected_category_id, first)
                    self.assertIs(app.runner, runner)
                    self.assertEqual(app.runner.queue, ["first", "second"])
                    self.assertEqual(app.playback_status, "Playing")
                    app.playback.adjust_volume.assert_not_called()
                    await pilot.press("escape")

    async def test_export_runs_in_background_without_changing_player_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            output = Path(tmp) / "export"
            started = threading.Event()
            release = threading.Event()

            def slow_export(path: Path) -> Path:
                started.set()
                release.wait(timeout=5)
                path.mkdir(parents=True)
                return path

            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True), patch(
                "tonepath.tui_privacy.default_privacy_export_path",
                return_value=output,
            ), patch("tonepath.tui_privacy.export_personal_data", side_effect=slow_export):
                config.write_config(config.default_config())
                app = TonepathApp()
                async with app.run_test() as pilot:
                    runner = SimpleNamespace(queue=["current"])
                    app.runner = runner
                    app.playback_status = "Playing"
                    screen = await self.open_privacy(app, pilot)
                    await pilot.press("e")
                    await self.wait_until(pilot, started.is_set)

                    self.assertEqual(screen.busy, "export")
                    await pilot.press("j")
                    self.assertIs(app.runner, runner)
                    self.assertEqual(app.playback_status, "Playing")

                    release.set()
                    await self.wait_until(pilot, lambda: not screen.busy)
                    self.assertTrue(output.exists())
                    self.assertIn(str(output), screen.status_message)
                    await pilot.press("escape")

    async def test_delete_modal_requires_exact_lowercase_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            memory = home / "memory" / "profile.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("private", encoding="utf-8")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await self.open_privacy(app, pilot)
                    await pilot.press("d")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyDeleteModal))
                    modal = app.screen
                    confirmation = modal.query_one("#privacy-confirm-input", Input)
                    confirmation.value = "DELETE"
                    await pilot.press("enter")

                    self.assertIsInstance(app.screen, PrivacyDeleteModal)
                    self.assertTrue(memory.exists())
                    self.assertIn("lowercase delete", modal.query_one("#privacy-confirm-status").render().plain)

                    confirmation.value = "delete"
                    await pilot.press("enter")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyScreen))
                    screen = app.screen
                    await self.wait_until(pilot, lambda: not screen.busy)
                    self.assertFalse(memory.exists())
                    await pilot.press("escape")

    async def test_confirmed_delete_keeps_screen_open_but_does_not_block_player_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            memory = home / "memory" / "profile.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("private", encoding="utf-8")
            started = threading.Event()
            release = threading.Event()

            def slow_delete(plan: object) -> PrivacyDeleteResult:
                started.set()
                release.wait(timeout=5)
                return PrivacyDeleteResult(
                    schema="tonepath-privacy-delete-result-v1",
                    plan_fingerprint=getattr(plan, "fingerprint"),
                    items=(
                        PrivacyDeleteItemResult(
                            "memory",
                            str(memory),
                            "deleted",
                            "Removed from Tonepath active storage.",
                        ),
                    ),
                )

            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True), patch(
                "tonepath.tui_privacy.execute_privacy_delete",
                side_effect=slow_delete,
            ):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    screen = await self.open_privacy(app, pilot)
                    original_mode = app.playback_mode
                    await pilot.press("d")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyDeleteModal))
                    app.screen.query_one("#privacy-confirm-input", Input).value = "delete"
                    await pilot.press("enter")
                    await self.wait_until(pilot, started.is_set)

                    self.assertEqual(screen.busy, "delete")
                    await pilot.press("m")
                    self.assertNotEqual(app.playback_mode, original_mode)
                    await pilot.press("escape")
                    self.assertIs(app.screen, screen)

                    release.set()
                    await self.wait_until(pilot, lambda: not screen.busy)
                    await pilot.press("escape")
                    self.assertIsNot(app.screen, screen)

    async def test_memory_delete_keeps_current_queue_and_clears_memory_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            memory = home / "memory" / "profile.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("private", encoding="utf-8")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    runner = runner_stub(["current"])
                    app.runner = runner
                    app.playback_status = "Playing"
                    app.memory_draft = "unsaved private draft"
                    await self.open_privacy(app, pilot)

                    await self.confirm_delete(app, pilot)

                    self.assertIs(app.runner, runner)
                    self.assertEqual(app.runner.queue, ["current"])
                    self.assertEqual(app.playback_status, "Playing")
                    self.assertEqual(app.memory_draft, "")
                    await pilot.press("escape")

    async def test_stale_plan_requires_a_fresh_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            memory = home / "memory" / "profile.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("before", encoding="utf-8")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await self.open_privacy(app, pilot)
                    await pilot.press("d")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyDeleteModal))
                    memory.write_text("changed after preview", encoding="utf-8")
                    modal = app.screen
                    modal.query_one("#privacy-confirm-input", Input).value = "delete"
                    await pilot.press("enter")
                    await self.wait_until(pilot, lambda: isinstance(app.screen, PrivacyScreen))
                    screen = app.screen
                    await self.wait_until(pilot, lambda: not screen.busy)

                    self.assertTrue(memory.exists())
                    self.assertIn("changed since the preview", screen.status_message)
                    self.assertIn("confirm again", screen.status_message)
                    await pilot.press("escape")

    async def test_history_delete_stops_playback_and_clears_runner_after_db_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = TonepathStore(home / "tonepath.db")
                request = SessionRequest("focus", "irritated", "focus", 1800)
                store.save_session(SessionPlan(request, tuple(build_phases(request))))
                store.close()
                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.runner = SimpleNamespace(queue=["current"])
                    app.playback = MagicMock()
                    app.playback_status = "Playing"
                    screen = await self.open_privacy(app, pilot)
                    await pilot.press("j", "j")
                    self.assertEqual(screen.selected_category_id, "history")

                    await self.confirm_delete(app, pilot)

                    self.assertIsNone(app.runner)
                    self.assertEqual(app.playback_status, "Ready")
                    app.playback.stop_current.assert_called_once()
                    self.assertEqual(app.store.profile_summary()["sessions"], 0)
                    await pilot.press("escape")

    async def test_all_personal_delete_clears_personal_state_and_current_history_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            memory = home / "memory" / "profile.md"
            memory.parent.mkdir(parents=True)
            memory.write_text("private", encoding="utf-8")
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.runner = runner_stub(["current"])
                    app.playback = MagicMock()
                    app.playback_status = "Playing"
                    app.memory_draft = "unsaved private draft"
                    app.memory_suggestions = [{"id": "pending"}]
                    app.store.record_feedback("like")
                    screen = await self.open_privacy(app, pilot)
                    await pilot.press("j", "j", "j", "j", "j")
                    self.assertEqual(screen.selected_category_id, "all-personal")

                    await self.confirm_delete(app, pilot)

                    self.assertFalse(memory.exists())
                    self.assertEqual(app.memory_draft, "")
                    self.assertEqual(app.memory_suggestions, [])
                    self.assertIsNone(app.runner)
                    self.assertEqual(app.playback_status, "Ready")
                    app.playback.stop_current.assert_called_once()
                    self.assertEqual(app.store.profile_summary()["feedback"], 0)
                    await pilot.press("escape")

    async def test_read_only_category_does_not_open_delete_modal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                config.write_config(config.default_config())
                app = TonepathApp()
                async with app.run_test() as pilot:
                    screen = await self.open_privacy(app, pilot)
                    await pilot.press("j", "j", "j")
                    self.assertEqual(screen.selected_category_id, "library-evidence")
                    await pilot.press("d")
                    await pilot.pause()

                    self.assertIs(app.screen, screen)
                    self.assertIn("read-only", screen.status_message)
                    await pilot.press("escape")

    async def test_partial_history_file_delete_does_not_reset_active_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    runner = runner_stub(["current"])
                    app.runner = runner
                    app.playback = MagicMock()
                    result = PrivacyDeleteResult(
                        schema="tonepath-privacy-delete-result-v1",
                        plan_fingerprint="fingerprint",
                        items=(
                            PrivacyDeleteItemResult(
                                "history",
                                str(home / "cache" / "audit"),
                                "deleted",
                                "Removed audit cache.",
                            ),
                            PrivacyDeleteItemResult(
                                "history",
                                "SQLite history records",
                                "failed",
                                "database locked",
                            ),
                        ),
                    )

                    app.on_privacy_data_deleted(PrivacyDataDeleted(result))

                    self.assertIs(app.runner, runner)
                    app.playback.stop_current.assert_not_called()
                    await pilot.press("ctrl+q")

    async def test_completed_absent_memory_delete_clears_only_memory_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                app = TonepathApp()
                async with app.run_test() as pilot:
                    runner = runner_stub(["current"])
                    app.runner = runner
                    app.memory_draft = "unsaved private draft"
                    app.memory_status_message = "old status"
                    result = PrivacyDeleteResult(
                        schema="tonepath-privacy-delete-result-v1",
                        plan_fingerprint="fingerprint",
                        items=(
                            PrivacyDeleteItemResult(
                                "memory",
                                "SQLite memory records",
                                "already_absent",
                                "No matching records were present.",
                            ),
                            PrivacyDeleteItemResult(
                                "memory",
                                str(home / "memory"),
                                "already_absent",
                                "Path is not present.",
                            ),
                            PrivacyDeleteItemResult(
                                "memory",
                                str(home / "cache" / "memory"),
                                "already_absent",
                                "Path is not present.",
                            ),
                        ),
                    )

                    app.on_privacy_data_deleted(PrivacyDataDeleted(result))

                    self.assertEqual(app.memory_draft, "")
                    self.assertIn("deleted", app.memory_status_message)
                    self.assertIs(app.runner, runner)
                    await pilot.press("ctrl+q")

    def test_delete_is_blocked_while_related_background_writers_are_active(self) -> None:
        memory_app = SimpleNamespace(memory_busy=True, request_busy=False)
        request_app = SimpleNamespace(memory_busy=False, request_busy=True)
        preparation_app = SimpleNamespace(memory_busy=False, request_busy=False, setup_prepare_busy=True)

        self.assertIn("Memory learning", deletion_task_blocker(memory_app, ("memory",)))
        self.assertIn("Memory learning", deletion_task_blocker(memory_app, ("personalization",)))
        self.assertIn("Request planning", deletion_task_blocker(request_app, ("history",)))
        self.assertIn("preparation is still running", deletion_task_blocker(preparation_app, ("history",)))
        self.assertIsNone(deletion_task_blocker(memory_app, ("history",)))

    def test_category_delete_completed_requires_no_failed_components(self) -> None:
        complete = PrivacyDeleteResult(
            schema="tonepath-privacy-delete-result-v1",
            plan_fingerprint="fingerprint",
            items=(
                PrivacyDeleteItemResult("memory", "database", "already_absent", "none"),
                PrivacyDeleteItemResult("memory", "files", "deleted", "removed"),
            ),
        )
        partial = PrivacyDeleteResult(
            schema="tonepath-privacy-delete-result-v1",
            plan_fingerprint="fingerprint",
            items=(
                PrivacyDeleteItemResult("memory", "database", "deleted", "removed"),
                PrivacyDeleteItemResult("memory", "files", "failed", "permission denied"),
            ),
        )

        self.assertTrue(category_delete_completed(complete, "memory"))
        self.assertFalse(category_delete_completed(partial, "memory"))
        self.assertFalse(category_delete_completed(complete, "history"))


def runner_stub(queue: list[str]) -> MagicMock:
    runner = MagicMock()
    runner.queue = queue
    runner.current.return_value = None
    runner.upcoming.return_value = []
    runner.active_plan.return_value = plan_session("focus for 20 minutes")
    return runner


if __name__ == "__main__":
    unittest.main()
