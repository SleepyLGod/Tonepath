import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.history import list_history, prepare_replay
from tonepath.models import CandidateScore, SessionPlan, Track, TrackFeatures
from tonepath.playback import MpvAdapter
from tonepath.planner import plan_session
from tonepath.tui import TonepathApp
from tonepath.tui_history import HistoryScreen, ReplaySessionRunner, history_replay_status


class FakeProcess:
    pid = 12345

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        return None


class ReplaySessionRunnerTest(unittest.TestCase):
    def test_replay_runner_uses_snapshot_queue_without_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            plan = plan_session("focus for 30 minutes")
            track_ids = [
                self.add_track(store, Path(tmp) / "first.mp3"),
                self.add_track(store, Path(tmp) / "second.mp3"),
            ]
            source_session_id = store.save_session(plan)
            source_queue = [
                CandidateScore(
                    track=store.get_track(track_id),
                    phase=plan.phases[index % len(plan.phases)],
                    score=2.0 - index,
                    confidence="high",
                    reasons=(f"snapshot reason {index}",),
                )
                for index, track_id in enumerate(track_ids)
            ]
            store.replace_session_queue(source_session_id, source_queue)
            replay = prepare_replay(store, source_session_id)

            with patch("tonepath.session.select_path", side_effect=AssertionError("selector called")):
                runner = ReplaySessionRunner(store, replay)

            self.assertNotEqual(runner.session_id, source_session_id)
            self.assertEqual(
                [candidate.track.path.name for candidate in runner.queue],
                ["first.mp3", "second.mp3"],
            )
            self.assertEqual(
                [Path(str(row["track_path"])).name for row in store.session_queue_items(runner.session_id)],
                ["first.mp3", "second.mp3"],
            )
            store.close()

    @staticmethod
    def add_track(store: TonepathStore, path: Path) -> int:
        path.write_bytes(b"not real audio")
        return store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash=path.name,
                mtime=1.0,
                title=path.stem,
                artist="artist",
                album=None,
                genre=None,
                duration=120.0,
                format="mp3",
            )
        )


class HistoryReplayStatusTest(unittest.TestCase):
    def test_replay_status_distinguishes_ready_partial_unavailable_and_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            plan = plan_session("focus for 30 minutes")

            legacy_id = store.save_session(plan)

            ready_track_id = ReplaySessionRunnerTest.add_track(store, Path(tmp) / "ready.mp3")
            ready_id = store.save_session(plan)
            store.replace_session_queue(
                ready_id,
                [
                    CandidateScore(
                        track=store.get_track(ready_track_id),
                        phase=plan.phases[0],
                        score=1.0,
                        confidence="high",
                        reasons=("ready",),
                    )
                ],
            )

            partial_track_id = ReplaySessionRunnerTest.add_track(store, Path(tmp) / "partial.mp3")
            partial_missing_id = ReplaySessionRunnerTest.add_track(
                store,
                Path(tmp) / "partial-missing.mp3",
            )
            partial_id = store.save_session(plan)
            store.replace_session_queue(
                partial_id,
                [
                    CandidateScore(
                        track=store.get_track(partial_track_id),
                        phase=plan.phases[0],
                        score=1.0,
                        confidence="high",
                        reasons=("partial",),
                    ),
                    CandidateScore(
                        track=store.get_track(partial_missing_id),
                        phase=plan.phases[0],
                        score=0.5,
                        confidence="high",
                        reasons=("missing",),
                    ),
                ],
            )
            (Path(tmp) / "partial-missing.mp3").rename(Path(tmp) / "partial-missing.gone")

            unavailable_track_id = ReplaySessionRunnerTest.add_track(
                store,
                Path(tmp) / "unavailable.mp3",
            )
            unavailable_id = store.save_session(plan)
            store.replace_session_queue(
                unavailable_id,
                [
                    CandidateScore(
                        track=store.get_track(unavailable_track_id),
                        phase=plan.phases[0],
                        score=1.0,
                        confidence="high",
                        reasons=("unavailable",),
                    )
                ],
            )
            (Path(tmp) / "unavailable.mp3").rename(Path(tmp) / "unavailable.gone")

            sessions = {
                session.id: session for session in list_history(store, include_all=True)
            }

            self.assertEqual(history_replay_status(store, sessions[ready_id]), "Ready")
            self.assertEqual(history_replay_status(store, sessions[partial_id]), "Partial")
            self.assertEqual(
                history_replay_status(store, sessions[unavailable_id]),
                "Unavailable",
            )
            self.assertEqual(history_replay_status(store, sessions[legacy_id]), "Legacy")
            store.close()


class HistoryScreenTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.ipc_ready = patch.object(MpvAdapter, "wait_for_ipc")
        self.ipc_ready.start()
        self.addCleanup(self.ipc_ready.stop)
        properties: dict[str, object] = {
            "pause": False,
            "time-pos": 12.0,
            "duration": 180.0,
            "volume": 100.0,
        }

        def send_command(_adapter: MpvAdapter, _ipc_path: Path, command: list[object]) -> object:
            if command[0] == "get_property":
                return properties[str(command[1])]
            if command[0] == "set_property":
                properties[str(command[1])] = command[2]
                return None
            if command[0] == "seek":
                return None
            raise AssertionError(f"Unexpected mpv command: {command}")

        self.ipc_commands = patch.object(MpvAdapter, "send_command", autospec=True, side_effect=send_command)
        self.ipc_commands.start()
        self.addCleanup(self.ipc_commands.stop)

    async def wait_for_request_idle(self, app: TonepathApp, pilot: object) -> None:
        for _ in range(80):
            if not app.request_busy:
                return
            await pilot.pause(0.05)
        self.fail("request planning worker did not finish")

    async def test_ctrl_l_browses_history_without_changing_current_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")

                async with app.run_test() as pilot:
                    runner = app.runner
                    queue = list(runner.queue) if runner is not None else []
                    await pilot.press("ctrl+l")

                    self.assertIsInstance(app.screen, HistoryScreen)
                    screen = app.screen
                    self.assertEqual(len(screen.sessions), 1)
                    self.assertIs(app.runner, runner)
                    self.assertEqual(app.runner.queue if app.runner is not None else [], queue)

                    await pilot.press("ctrl+l")
                    self.assertNotIsInstance(app.screen, HistoryScreen)
                    self.assertIs(app.runner, runner)
                    self.assertEqual(app.runner.queue if app.runner is not None else [], queue)
                    self.assertIn("Ctrl+L", app.command_bar_renderable().plain)
                    self.assertIn("listening history", app.help_panel_text())
                    await pilot.press("ctrl+q")

    async def test_history_selection_refreshes_and_only_lists_played_or_saved_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                played_id = self.seed_history(tmp)
                store = TonepathStore()
                track_id = store.list_tracks()[0].id
                plan = plan_session("hidden path for 30 minutes")
                hidden_id = store.save_session(plan)
                saved_id = store.save_session(plan_session("saved calm path for 20 minutes"))
                store.replace_session_queue(
                    saved_id,
                    [
                        CandidateScore(
                            track=store.get_track(track_id),
                            phase=plan.phases[0],
                            score=1.0,
                            confidence="high",
                            reasons=("saved",),
                        )
                    ],
                )
                store.save_session_bookmark(saved_id, "Bedtime")
                store.close()
                app = TonepathApp()

                async with app.run_test() as pilot:
                    await pilot.press("ctrl+l")
                    self.assertIsInstance(app.screen, HistoryScreen)
                    screen = app.screen
                    self.assertEqual({session.id for session in screen.sessions}, {played_id, saved_id})
                    self.assertNotIn(hidden_id, {session.id for session in screen.sessions})
                    first_id = screen.selected_session_id
                    await pilot.press("down")
                    self.assertNotEqual(screen.selected_session_id, first_id)
                    await pilot.press("up")
                    self.assertEqual(screen.selected_session_id, first_id)
                    await pilot.press("escape")
                    await pilot.press("ctrl+q")

    async def test_history_browsing_keeps_playback_running_and_load_stops_without_autoplay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start, patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        self.assertEqual(start.call_count, 1)
                        await pilot.press("ctrl+l")
                        self.assertIsInstance(app.screen, HistoryScreen)
                        self.assertEqual(app.playback_status, "Playing")
                        self.assertEqual(stop.call_count, 0)
                        pulse_before_poll = app.pulse_tick
                        app.poll_playback_finished()
                        for _ in range(20):
                            if not app.playback_state_busy:
                                break
                            await pilot.pause(0.01)
                        self.assertIsInstance(app.screen, HistoryScreen)
                        self.assertGreater(app.pulse_tick, pulse_before_poll)

                        await pilot.press("enter")
                        await pilot.pause()
                        self.assertNotIsInstance(app.screen, HistoryScreen)
                        self.assertEqual(app.playback_status, "Ready")
                        self.assertEqual(stop.call_count, 1)
                        self.assertEqual(start.call_count, 1)
                        await pilot.press("ctrl+q")

    async def test_smart_request_planning_keeps_current_playback_responsive_until_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            env = {
                "TONEPATH_HOME": str(home),
                "TONEPATH_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "secret",
            }
            with patch.dict(os.environ, env, clear=True):
                self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")
                started = threading.Event()
                release = threading.Event()
                call_count = 0

                def slow_plan(prompt: str, _settings: config.TonepathConfig) -> tuple[SessionPlan, str]:
                    nonlocal call_count
                    call_count += 1
                    started.set()
                    release.wait(2)
                    return plan_session(prompt), "LLM intent: parsed with deepseek."

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start, patch.object(
                    MpvAdapter,
                    "stop_process",
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        runner = app.runner
                        queue = list(runner.queue) if runner is not None else []
                        config.write_config(config.preset_config("smart", send_to_llm=True))
                        self.configure_music_dir(tmp)
                        with patch("tonepath.tui.smart_plan_session", side_effect=slow_plan):
                            prompt_input = app.query_one("#prompt-input")
                            prompt_input.value = "new calm focus path for 20 minutes"
                            prompt_input.focus()
                            await pilot.press("enter")
                            for _ in range(40):
                                if started.is_set():
                                    break
                                await pilot.pause(0.05)
                            self.assertTrue(started.is_set())
                            self.assertTrue(app.request_busy)
                            self.assertIs(app.runner, runner)
                            self.assertEqual(app.runner.queue if app.runner is not None else [], queue)
                            self.assertEqual(app.playback_status, "Playing")
                            self.assertEqual(stop.call_count, 0)
                            status = app.query_one("#status-bar").render()
                            self.assertIn("Planning next path", status.plain)
                            app.start_request_planning("duplicate request")
                            self.assertEqual(call_count, 1)
                            await pilot.press("m")
                            self.assertEqual(app.playback_mode, "Continue Path")
                            release.set()
                            await self.wait_for_request_idle(app, pilot)

                        self.assertIsNot(app.runner, runner)
                        self.assertEqual(app.runner.prompt, "new calm focus path for 20 minutes")
                        self.assertEqual(app.playback_status, "Ready")
                        self.assertEqual(stop.call_count, 1)
                        self.assertEqual(start.call_count, 1)
                        await pilot.press("ctrl+q")

    async def test_failed_smart_request_planning_preserves_current_path_and_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            env = {
                "TONEPATH_HOME": str(home),
                "TONEPATH_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "secret",
            }
            with patch.dict(os.environ, env, clear=True):
                self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter,
                    "stop_process",
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        runner = app.runner
                        queue = list(runner.queue) if runner is not None else []
                        config.write_config(config.preset_config("smart", send_to_llm=True))
                        self.configure_music_dir(tmp)
                        with patch(
                            "tonepath.tui.smart_plan_session",
                            side_effect=RuntimeError("provider unavailable"),
                        ):
                            app.start_request_planning("new path that fails")
                            await self.wait_for_request_idle(app, pilot)

                        self.assertIs(app.runner, runner)
                        self.assertEqual(app.runner.queue if app.runner is not None else [], queue)
                        self.assertEqual(app.playback_status, "Playing")
                        self.assertEqual(stop.call_count, 0)
                        self.assertIn("provider unavailable", app.request_status_message)
                        await pilot.press("ctrl+q")

    async def test_failed_request_activation_preserves_current_path_and_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            env = {
                "TONEPATH_HOME": str(home),
                "TONEPATH_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "secret",
            }
            with patch.dict(os.environ, env, clear=True):
                self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter,
                    "stop_process",
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        runner = app.runner
                        queue = list(runner.queue) if runner is not None else []
                        config.write_config(config.preset_config("smart", send_to_llm=True))
                        self.configure_music_dir(tmp)
                        with patch(
                            "tonepath.tui.smart_plan_session",
                            return_value=(
                                plan_session("replacement path for 20 minutes"),
                                "LLM intent: parsed with deepseek.",
                            ),
                        ), patch(
                            "tonepath.tui.SessionRunner",
                            side_effect=RuntimeError("selector failed"),
                        ):
                            app.start_request_planning("replacement path for 20 minutes")
                            await self.wait_for_request_idle(app, pilot)

                        self.assertIs(app.runner, runner)
                        self.assertEqual(app.runner.queue if app.runner is not None else [], queue)
                        self.assertEqual(app.playback_status, "Playing")
                        self.assertEqual(stop.call_count, 0)
                        self.assertIn("selector failed", app.request_status_message)
                        await pilot.press("ctrl+q")

    async def test_enter_loads_exact_history_queue_without_autoplay_or_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                source_session_id = self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        app.intent_note = "Intent parsed for the previous Request."
                        with patch.object(app, "log_event", wraps=app.log_event) as log_event:
                            await pilot.press("ctrl+l")
                            with patch("tonepath.session.select_path", side_effect=AssertionError("selector called")):
                                await pilot.press("enter")
                                await pilot.pause()

                        self.assertNotIsInstance(app.screen, HistoryScreen)
                        self.assertIsNotNone(app.runner)
                        self.assertNotEqual(app.runner.session_id, source_session_id)
                        self.assertEqual(
                            [candidate.track.path.name for candidate in app.runner.queue],
                            ["history-first.mp3", "history-second.mp3"],
                        )
                        self.assertEqual(app.playback_status, "Ready")
                        self.assertIsNone(app.intent_note)
                        self.assertEqual(start.call_count, 0)
                        self.assertTrue(
                            any(
                                call.args
                                and call.args[0]
                                == f"Loaded exact history path from session {source_session_id}. Press Space to play."
                                for call in log_event.call_args_list
                            )
                        )
                        await pilot.press("ctrl+q")

    async def test_enter_reruns_legacy_request_without_autoplay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                self.seed_history(tmp)
                store = TonepathStore()
                legacy_id = store.save_session(plan_session("legacy path for 20 minutes"))
                store.save_session_bookmark(legacy_id, "Legacy")
                store.close()
                app = TonepathApp("focus for 30 minutes")

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        runner = app.runner
                        queue = list(runner.queue) if runner is not None else []
                        store = app.store
                        self.assertIsNotNone(store)
                        sessions_before = store.conn.execute(
                            "SELECT COUNT(*) AS count FROM sessions"
                        ).fetchone()["count"]
                        await pilot.press("ctrl+l")
                        self.assertIsInstance(app.screen, HistoryScreen)
                        screen = app.screen
                        self.assertEqual(screen.selected_session_id, legacy_id)
                        self.assertEqual(screen.replay_statuses[legacy_id], "Legacy")
                        table = screen.query_one("#history-list")
                        column_labels = [column.label.plain for column in table.columns.values()]
                        self.assertEqual(column_labels[:2], ["Saved", "Replay"])
                        details = screen.query_one("#history-details").render()
                        self.assertIn("Legacy · Exact replay unavailable", details.plain)
                        self.assertIn("Press Enter to run this Request again", details.plain)
                        await pilot.press("space")
                        await pilot.pause()
                        self.assertIsInstance(app.screen, HistoryScreen)
                        self.assertIn("Space does not act in History", screen.status_message)
                        details = screen.query_one("#history-details").render()
                        self.assertIn("Status", details.plain)
                        self.assertNotIn("Cannot load", details.plain)
                        self.assertIs(app.runner, runner)
                        self.assertEqual(app.runner.queue if app.runner is not None else [], queue)
                        with patch.object(
                            app,
                            "start_request_planning",
                            wraps=app.start_request_planning,
                        ) as start_planning:
                            await pilot.press("enter")
                            await pilot.pause()

                        self.assertNotIsInstance(app.screen, HistoryScreen)
                        start_planning.assert_called_once_with(
                            "legacy path for 20 minutes",
                            history_source_session_id=legacy_id,
                        )
                        self.assertIsNot(app.runner, runner)
                        self.assertEqual(app.runner.prompt, "legacy path for 20 minutes")
                        self.assertTrue(app.runner.queue)
                        self.assertEqual(app.playback_status, "Ready")
                        self.assertEqual(start.call_count, 0)
                        sessions_after = store.conn.execute(
                            "SELECT COUNT(*) AS count FROM sessions"
                        ).fetchone()["count"]
                        self.assertEqual(sessions_after, sessions_before + 1)
                        await pilot.press("ctrl+q")

    async def test_r_reruns_ready_history_request_instead_of_exact_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                source_session_id = self.seed_history(tmp)
                app = TonepathApp("focus for 30 minutes")

                async with app.run_test() as pilot:
                    store = app.store
                    self.assertIsNotNone(store)
                    sessions_before = store.conn.execute(
                        "SELECT COUNT(*) AS count FROM sessions"
                    ).fetchone()["count"]
                    await pilot.press("ctrl+l")
                    self.assertIsInstance(app.screen, HistoryScreen)
                    screen = app.screen
                    self.assertEqual(screen.selected_session_id, source_session_id)
                    await pilot.press("r")
                    await pilot.pause()

                    self.assertNotIsInstance(app.screen, HistoryScreen)
                    self.assertEqual(app.runner.prompt, "saved focus path for 30 minutes")
                    self.assertEqual(app.playback_status, "Ready")
                    sessions_after = store.conn.execute(
                        "SELECT COUNT(*) AS count FROM sessions"
                    ).fetchone()["count"]
                    self.assertEqual(sessions_after, sessions_before + 1)
                    await pilot.press("ctrl+q")

    async def test_history_load_omits_missing_files_and_reports_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                self.seed_history(tmp)
                store = TonepathStore()
                tracks = store.list_tracks()
                missing_id = self.add_ready_track(store, Path(tmp) / "missing.mp3")
                plan = plan_session("mixed history path for 20 minutes")
                session_id = store.save_session(plan)
                store.replace_session_queue(
                    session_id,
                    [
                        self.candidate(store, tracks[0].id, plan, 0),
                        self.candidate(store, missing_id, plan, 1),
                        self.candidate(store, tracks[1].id, plan, 2),
                    ],
                )
                play_id = store.start_play(session_id, tracks[0].id)
                store.end_play(play_id)
                (Path(tmp) / "missing.mp3").rename(Path(tmp) / "missing.gone")
                store.conn.execute("DELETE FROM tracks WHERE id = ?", (missing_id,))
                store.conn.commit()
                store.close()
                app = TonepathApp("focus for 30 minutes")

                async with app.run_test() as pilot:
                    with patch.object(app, "log_event", wraps=app.log_event) as log_event:
                        await pilot.press("ctrl+l")
                        await pilot.press("enter")
                        await pilot.pause()

                    self.assertNotIsInstance(app.screen, HistoryScreen)
                    self.assertEqual(
                        [candidate.track.path.name for candidate in app.runner.queue],
                        [tracks[0].path.name, tracks[1].path.name],
                    )
                    self.assertTrue(
                        any(
                            call.args
                            and call.args[0] == "History omitted missing file: missing"
                            for call in log_event.call_args_list
                        )
                    )
                    await pilot.press("ctrl+q")

    async def test_enter_reruns_unavailable_history_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                self.seed_history(tmp)
                store = TonepathStore()
                missing_id = self.add_ready_track(store, Path(tmp) / "only-missing.mp3")
                plan = plan_session("missing history path for 20 minutes")
                session_id = store.save_session(plan)
                store.replace_session_queue(
                    session_id,
                    [self.candidate(store, missing_id, plan, 0)],
                )
                store.save_session_bookmark(session_id, "Unavailable path")
                play_id = store.start_play(session_id, missing_id)
                store.end_play(play_id)
                (Path(tmp) / "only-missing.mp3").rename(Path(tmp) / "only-missing.gone")
                store.conn.execute("DELETE FROM tracks WHERE id = ?", (missing_id,))
                store.conn.commit()
                store.close()
                app = TonepathApp("focus for 30 minutes")

                async with app.run_test() as pilot:
                    runner = app.runner
                    await pilot.press("ctrl+l")
                    screen = app.screen
                    self.assertEqual(screen.selected_session_id, session_id)
                    self.assertEqual(screen.replay_statuses[session_id], "Unavailable")
                    await pilot.press("enter")
                    await pilot.pause()

                    self.assertNotIsInstance(app.screen, HistoryScreen)
                    self.assertIsNot(app.runner, runner)
                    self.assertEqual(app.runner.prompt, "missing history path for 20 minutes")
                    self.assertTrue(app.runner.queue)
                    self.assertEqual(app.playback_status, "Ready")
                    await pilot.press("ctrl+q")

    def seed_history(self, tmp: str) -> int:
        store = TonepathStore()
        self.configure_music_dir(tmp)
        track_ids = [
            self.add_ready_track(store, Path(tmp) / "history-first.mp3"),
            self.add_ready_track(store, Path(tmp) / "history-second.mp3"),
            self.add_ready_track(store, Path(tmp) / "current.mp3"),
        ]
        plan = plan_session("saved focus path for 30 minutes")
        session_id = store.save_session(plan)
        queue = [
            CandidateScore(
                track=store.get_track(track_id),
                phase=plan.phases[index % len(plan.phases)],
                score=3.0 - index,
                confidence="high",
                reasons=(f"history reason {index}",),
            )
            for index, track_id in enumerate(track_ids[:2])
        ]
        store.replace_session_queue(session_id, queue)
        play_id = store.start_play(session_id, track_ids[0])
        store.end_play(play_id)
        store.close()
        return session_id

    @staticmethod
    def candidate(
        store: TonepathStore,
        track_id: int,
        plan: SessionPlan,
        index: int,
    ) -> CandidateScore:
        return CandidateScore(
            track=store.get_track(track_id),
            phase=plan.phases[index % len(plan.phases)],
            score=3.0 - index,
            confidence="high",
            reasons=(f"history reason {index}",),
        )

    @staticmethod
    def add_ready_track(store: TonepathStore, path: Path) -> int:
        track_id = ReplaySessionRunnerTest.add_track(store, path)
        store.upsert_features(
            TrackFeatures(
                track_id=track_id,
                bpm=100.0,
                loudness=-14.0,
                energy=0.4,
                vocalness=0.2,
                feature_source="test",
                confidence="high",
            )
        )
        return track_id

    @staticmethod
    def configure_music_dir(tmp: str) -> None:
        current = config.load_config()
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
            )
        )


if __name__ == "__main__":
    unittest.main()
