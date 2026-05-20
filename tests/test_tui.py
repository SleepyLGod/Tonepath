import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.playback import MpvAdapter
from tonepath.tui import TonepathApp, bpm_text, confidence_label, energy_meter, queue_marker, vocalness_text
from textual.widgets import Input


class FakeProcess:
    pid = 9876

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


class FinishedProcess(FakeProcess):
    def poll(self) -> int | None:
        return 0


class TonepathTuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_tui_launches_intake_without_session_or_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp()
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        self.assertIsNotNone(app.query_one("#prompt-input", Input))
                        self.assertIsNone(app.runner)
                        await pilot.press("space")
                        await pilot.press("s")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 0)

    async def test_tui_prompt_submit_creates_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt_input = app.query_one("#prompt-input", Input)
                    prompt_input.value = "我现在很烦，想半小时后进入写代码状态，不要人声"
                    await pilot.press("enter")
                    self.assertIsNotNone(app.runner)
                    self.assertIn("irritated", app.timeline_text())
                    await pilot.press("q")

    async def test_tui_codex_keys_do_not_run_background_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("a")
                        await pilot.press("r")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 0)

    async def test_tui_rerank_reads_latest_codex_result_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                home = Path(tmp) / "home"
                result_dir = home / "cache" / "audit" / "run-1"
                result_dir.mkdir(parents=True)
                (result_dir / "evidence.json").write_text(
                    json.dumps({"prompt": "from irritated to focus in 30 minutes"}),
                    encoding="utf-8",
                )
                (result_dir / "codex-result.json").write_text(
                    json.dumps(
                        {
                            "summary": "Codex reviewed the path.",
                            "decisions": [
                                {"decision": "keep"},
                                {"decision": "demote"},
                                {"decision": "reject"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    await pilot.press("r")
                    self.assertEqual(
                        app.latest_codex_summary(),
                        "Codex reviewed the path. keep 1 · demote 1 · reject 1",
                    )
                    await pilot.press("q")

    async def test_tui_rerank_ignores_codex_result_for_other_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                home = Path(tmp) / "home"
                result_dir = home / "cache" / "audit" / "run-1"
                result_dir.mkdir(parents=True)
                (result_dir / "evidence.json").write_text(
                    json.dumps({"prompt": "evening relaxation"}),
                    encoding="utf-8",
                )
                (result_dir / "codex-result.json").write_text(
                    json.dumps(
                        {
                            "summary": "Wrong prompt.",
                            "decisions": [{"decision": "keep"}],
                        }
                    ),
                    encoding="utf-8",
                )
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    await pilot.press("r")
                    self.assertIsNone(app.latest_codex_summary())
                    await pilot.press("q")

    async def test_tui_intake_guides_prepare_when_features_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    renderable = app.query_one("#now-playing").render()
                    self.assertIn("tonepath prepare", renderable.plain)
                    self.assertEqual(app.missing_feature_count(), 1)
                    await pilot.press("q")

    async def test_tui_intake_guides_model_setup_when_runtime_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = self.add_track(store, tmp, "a.mp3")
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        bpm=100.0,
                        loudness=-14.0,
                        energy=0.5,
                        feature_source="test",
                        confidence="medium",
                    )
                )
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    renderable = app.query_one("#now-playing").render()
                    self.assertIn("models setup essentia-tf", renderable.plain)
                    await pilot.press("q")

    async def test_tui_launches_session_screen_with_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        self.assertIsNotNone(app.query_one("#timeline"))
                        self.assertIsNotNone(app.query_one("#queue"))
                        self.assertIsNotNone(app.query_one("#why-panel"))
                        self.assertIsNotNone(app.query_one("#event-log"))
                        self.assertEqual(app.query_one("#queue").ordered_columns[3].label.plain, "Energy")
                        self.assertIn("Fit", app.why_panel_text())
                        self.assertIn("Evidence", app.why_panel_text())
                        self.assertIn("Unknown", app.why_panel_text())
                        self.assertIn("◇", app.timeline_text())
                        self.assertIn("✓ offline", app.privacy_text())
                        self.assertEqual(len(app.privacy_text().splitlines()), 3)
                        await pilot.press("w")
                        await pilot.press("s")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 0)

    async def test_tui_queue_energy_uses_features_or_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        energy=0.42,
                        loudness=-18.0,
                        feature_source="test",
                        confidence="medium",
                    )
                )
                store.close()

                app = TonepathApp("我现在很烦，想半小时后进入写代码状态，不要人声")
                async with app.run_test() as pilot:
                    self.assertEqual(app.energy_text(track_id), "0.42")
                    self.assertEqual(app.energy_text(None), "--")
                    self.assertEqual(bpm_text(None), "unknown")
                    self.assertEqual(bpm_text(118.0), "118")
                    self.assertEqual(vocalness_text(None), "unknown")
                    self.assertEqual(vocalness_text(0.24), "0.24")
                    self.assertEqual(energy_meter(None), "▯▯▯▯▯")
                    self.assertEqual(energy_meter(0.42), "▮▮▯▯▯")
                    self.assertEqual(queue_marker("now"), "▶")
                    self.assertEqual(queue_marker("+1"), "1")
                    self.assertEqual(confidence_label("medium"), "med")
                    await pilot.press("q")

    async def test_tui_play_starts_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 1)

    async def test_tui_quit_stops_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("q")
                self.assertTrue(stop.called)

    async def test_tui_skip_replaces_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start, patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("s")
                        await pilot.press("q")
                self.assertGreaterEqual(start.call_count, 2)
                self.assertTrue(stop.called)

    async def test_tui_stop_key_stops_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("p")
                        await pilot.press("x")
                        await pilot.press("q")
                self.assertTrue(stop.called)

    async def test_tui_natural_finish_records_play_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FinishedProcess()):
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        app.poll_playback_finished()
                        await pilot.press("q")

                store = TonepathStore()
                row = store.conn.execute("SELECT ended_at, skipped FROM plays").fetchone()
                self.assertIsNotNone(row["ended_at"])
                self.assertEqual(row["skipped"], 0)
                store.close()

    def add_track(self, store: TonepathStore, tmp: str, name: str) -> int:
        path = Path(tmp) / name
        path.write_bytes(b"not real audio")
        return store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash=name,
                mtime=1.0,
                title=name,
                artist="artist",
                album=None,
                genre=None,
                duration=None,
                format="mp3",
            )
        )


if __name__ == "__main__":
    unittest.main()
