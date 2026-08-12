import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input, TextArea

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.playback import MpvAdapter
from tonepath.tui import TonepathApp
from tonepath.tui_reactions import DislikedTracksScreen


class FakeProcess:
    pid = 2468

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


class TrackReactionTuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_like_and_dislike_toggle_without_changing_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                seed_home(tmp)
                app = TonepathApp("focus for 20 minutes")
                async with app.run_test() as pilot:
                    self.assertIsNotNone(app.runner)
                    current = app.runner.current()
                    self.assertIsNotNone(current)
                    track_id = current.track.id
                    original_queue = list(app.runner.queue)
                    original_index = app.runner.current_index

                    await pilot.press("u")

                    self.assertEqual(app.store.get_track_reaction(track_id), "disliked")
                    self.assertEqual(app.runner.queue, original_queue)
                    self.assertEqual(app.runner.current_index, original_index)
                    self.assertIn("Disliked", app.now_playing_text())
                    labels = [column.label.plain for column in app.query_one("#queue").columns.values()]
                    self.assertIn("You", labels)
                    current_row = app.query_one("#queue").get_row_at(0)
                    self.assertTrue(any(getattr(cell, "plain", str(cell)) == "×" for cell in current_row))

                    await pilot.press("u")
                    self.assertIsNone(app.store.get_track_reaction(track_id))
                    await pilot.press("l")
                    self.assertEqual(app.store.get_track_reaction(track_id), "liked")
                    self.assertIn("Liked", app.now_playing_text())
                    await pilot.press("ctrl+q")

    async def test_reaction_shortcuts_are_plain_text_in_request_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                seed_home(tmp)
                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt = app.query_one("#prompt-input", Input)
                    self.assertTrue(prompt.has_focus)
                    await pilot.press("h", "l", "u")
                    self.assertEqual(prompt.value, "hlu")
                    self.assertNotIsInstance(app.screen, DislikedTracksScreen)
                    await pilot.press("ctrl+q")

    async def test_reaction_shortcuts_are_plain_text_in_memory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                seed_home(tmp)
                app = TonepathApp()
                async with app.run_test() as pilot:
                    await pilot.press("ctrl+o")
                    memory = app.query_one("#memory-input", TextArea)
                    self.assertTrue(memory.has_focus)
                    await pilot.press("h", "l", "u")
                    self.assertEqual(memory.text, "hlu")
                    self.assertNotIsInstance(app.screen, DislikedTracksScreen)
                    await pilot.press("ctrl+q")

    async def test_disliked_screen_browses_previews_and_restores_without_changing_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                track_ids = seed_home(tmp)
                store = TonepathStore()
                hidden_id = track_ids[-1]
                store.set_track_reaction(hidden_id, "disliked")
                store.close()
                app = TonepathApp("focus for 20 minutes")

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start, patch.object(
                    MpvAdapter,
                    "wait_for_ipc",
                ), patch.object(MpvAdapter, "send_command", return_value=False), patch.object(
                    MpvAdapter,
                    "stop_process",
                ) as stop:
                    async with app.run_test() as pilot:
                        self.assertIsNotNone(app.runner)
                        runner = app.runner
                        queue = list(runner.queue)
                        session_count = app.store.profile_summary()["sessions"]
                        await pilot.press("space")
                        await pilot.press("h")

                        self.assertIsInstance(app.screen, DislikedTracksScreen)
                        self.assertEqual(app.runner.queue, queue)
                        stop.assert_not_called()

                        await pilot.press("space")
                        self.assertEqual(start.call_count, 2)
                        self.assertEqual(app.runner.queue, queue)
                        self.assertEqual(app.store.profile_summary()["sessions"], session_count)
                        self.assertEqual(app.reaction_preview_track_id, hidden_id)

                        await pilot.press("enter")
                        self.assertIsNone(app.store.get_track_reaction(hidden_id))
                        await pilot.press("escape")

                        self.assertNotIsInstance(app.screen, DislikedTracksScreen)
                        self.assertIs(app.runner, runner)
                        self.assertEqual(app.runner.queue, queue)
                        self.assertEqual(app.playback_status, "Ready")
                        self.assertGreaterEqual(stop.call_count, 2)
                        await pilot.press("ctrl+q")


def seed_home(tmp: str) -> list[int]:
    home = Path(os.environ["TONEPATH_HOME"])
    current = config.default_config()
    config.write_config(
        config.TonepathConfig(
            music_dirs=(tmp,),
            data_dir=current.data_dir,
            player=current.player,
            network_mode=current.network_mode,
            privacy=current.privacy,
            models=current.models,
            experience=current.experience,
            ui=current.ui,
            llm=current.llm,
        )
    )
    store = TonepathStore(home / "tonepath.db")
    track_ids: list[int] = []
    for index, name in enumerate(("first.mp3", "second.mp3", "hidden.mp3")):
        path = Path(tmp) / name
        path.write_bytes(b"audio")
        track_id = store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash=name,
                mtime=1.0,
                title=path.stem,
                artist="Artist",
                album=None,
                genre="instrumental",
                duration=120.0,
                format="mp3",
            )
        )
        store.upsert_features(
            TrackFeatures(
                track_id=track_id,
                bpm=96.0 + index,
                loudness=-16.0,
                energy=0.4,
                vocalness=0.1,
                feature_source="test",
                confidence="high",
            )
        )
        track_ids.append(track_id)
    store.close()
    return track_ids


if __name__ == "__main__":
    unittest.main()
