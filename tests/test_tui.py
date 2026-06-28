import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath import config
from tonepath.db import TonepathStore
from tonepath.models import CandidateScore, SessionPhase, Track, TrackFeatures
from tonepath.playback import MpvAdapter
from tonepath.planner import plan_session
from tonepath.profile import build_profile_evidence, deterministic_suggestions, save_suggestions
from tonepath.tui import (
    TonepathApp,
    bpm_text,
    confidence_label,
    energy_meter,
    fit_cell,
    format_clock,
    progress_bar,
    profile_learning_hint,
    pulse_meter,
    queue_marker,
    vocalness_text,
)
from tonepath.tui_theme import PALETTE_BY_KEY, PALETTES
from textual.theme import Theme
from textual.widgets import Input, Static


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
    def test_builtin_palettes_register_as_textual_themes(self) -> None:
        for palette in PALETTES:
            theme = Theme(
                name=palette.key,
                primary=palette.primary,
                secondary=palette.secondary,
                warning=palette.warning,
                success=palette.success,
                accent=palette.accent,
                foreground=palette.text,
                background=palette.background,
                surface=palette.surface,
                panel=palette.panel,
                dark=palette.dark,
            )
            self.assertEqual(theme.name, palette.key)

    def test_tui_profile_learning_hint_points_to_suggest_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                self.assertIn("profile suggest", profile_learning_hint())

    def test_tui_profile_learning_hint_points_to_inspect_when_suggestions_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = self.add_track(store, tmp, "a.mp3")
                store.upsert_features(
                    TrackFeatures(
                        track_id=track_id,
                        bpm=100.0,
                        loudness=-8.0,
                        energy=0.5,
                        vocalness=0.2,
                        feature_source="test",
                        confidence="medium",
                    )
                )
                session_id = store.save_session(plan_session("focus 30m"))
                store.record_feedback("too-loud", session_id=session_id, track_id=track_id)
                evidence = build_profile_evidence(store)
                save_suggestions(evidence, deterministic_suggestions(evidence), "deterministic")
                store.close()

                self.assertIn("profile inspect", profile_learning_hint())

    async def test_tui_launches_intake_without_session_or_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
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
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt_input = app.query_one("#prompt-input", Input)
                    prompt_input.value = "我现在很烦，想半小时后进入写代码状态，不要人声"
                    await pilot.press("enter")
                    self.assertIsNotNone(app.runner)
                    self.assertIn("irritated", app.timeline_text())
                    await pilot.press("q")

    async def test_tui_smart_mode_uses_smart_planner_fallback_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home), "TONEPATH_LLM_PROVIDER": "deepseek"}, clear=True):
                config.write_config(config.preset_config("smart"))
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt_input = app.query_one("#prompt-input", Input)
                    prompt_input.value = "focus 30m no vocals"
                    await pilot.press("enter")
                    self.assertIsNotNone(app.runner)
                    self.assertIn("DEEPSEEK_API_KEY missing", app.intent_note or "")
                    self.assertIn("✓ Smart", app.privacy_text())
                    self.assertIn("✓ AI Assist Missing Key: deepseek", app.privacy_text())
                    await pilot.press("q")

    async def test_tui_codex_keys_do_not_run_background_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("a")
                        await pilot.press("r")
                        await pilot.press("q")
                self.assertEqual(start.call_count, 0)

    async def test_tui_rerank_reads_latest_codex_result_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                home = Path(tmp) / "home"
                result_dir = home / "cache" / "audit" / "run-1"
                result_dir.mkdir(parents=True)
                (result_dir / "evidence.json").write_text(
                    json.dumps(
                        {
                            "run_id": "run-1",
                            "prompt": "from irritated to focus in 30 minutes",
                            "candidates": [
                                {"phase": "focus", "track": {"id": 1, "title": "Calm Track", "artist": "artist"}, "score": 1.0},
                                {"phase": "focus", "track": {"id": 2, "title": "Busy Track", "artist": "artist"}, "score": 0.8},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                (result_dir / "codex-result.json").write_text(
                    json.dumps(
                        {
                            "summary": "Codex reviewed the path.",
                            "decisions": [
                                {
                                    "track_id": 1,
                                    "decision": "keep",
                                    "fit_score": 0.9,
                                    "risk_flags": [],
                                    "reason": "local evidence fits",
                                    "evidence_used": [{"type": "local", "field": "bpm", "value": 90}],
                                },
                                {
                                    "track_id": 2,
                                    "decision": "demote",
                                    "fit_score": 0.4,
                                    "risk_flags": ["too stimulating"],
                                    "reason": "web and local evidence suggest demotion",
                                    "evidence_used": [{"type": "local", "field": "energy", "value": 0.8}],
                                },
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    await pilot.press("r")
                    self.assertEqual(
                        app.latest_codex_summary(),
                        "Rerank preview: keep 1 · demote 1 · reject 0 · not audited 0",
                    )
                    preview = app.latest_codex_preview()
                    self.assertIsNotNone(preview)
                    self.assertIn("keep: Calm Track - artist · keep", preview)
                    self.assertIn("demote: Busy Track - artist · move later", preview)
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
                self.add_ready_track(store, tmp, "a.mp3")
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

    async def test_tui_review_files_blocks_prompt_submit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt_input = app.query_one("#prompt-input", Input)
                    prompt_input.value = "我现在很烦，想半小时后进入写代码状态，不要人声"
                    await pilot.press("enter")
                    self.assertIsNone(app.runner)
                    renderable = app.query_one("#now-playing").render()
                    self.assertIn("Library is not ready", renderable.plain)
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
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start:
                    async with app.run_test() as pilot:
                        self.assertIsNotNone(app.query_one("#timeline"))
                        self.assertEqual(app.query_one("#timeline", Static).border_title, "Path")
                        self.assertIsNotNone(app.query_one("#queue"))
                        self.assertIsNotNone(app.query_one("#why-panel"))
                        self.assertIsNotNone(app.query_one("#event-log"))
                        self.assertIsNotNone(app.query_one("#command-bar"))
                        self.assertEqual(app.query_one("#queue").ordered_columns[3].label.plain, "Fit")
                        self.assertEqual(app.query_one("#queue").ordered_columns[4].label.plain, "Energy")
                        self.assertIn("Why", app.why_panel_text())
                        self.assertIn("Evidence", app.why_panel_text())
                        self.assertIn("Missing evidence", app.why_panel_text())
                        self.assertIn("◇", app.timeline_text())
                        self.assertIn("◇", app.timeline_renderable().plain)
                        self.assertIn("✓ Private", app.privacy_text())
                        self.assertIn("✓ AI Assist Off", app.privacy_text())
                        self.assertIn("✓ Model Missing", app.privacy_text())
                        self.assertIn("✓ Codex", app.privacy_text())
                        self.assertEqual(len(app.privacy_text().splitlines()), 5)
                        self.assertIn("Manual", app.status_bar_text())
                        self.assertNotIn("Mode Manual", app.status_bar_text())
                        self.assertFalse(app.status_bar_text().startswith(" "))
                        command_bar = app.command_bar_renderable().plain
                        self.assertIn("Space", command_bar)
                        self.assertIn("Play", command_bar)
                        self.assertIn(">  Next", command_bar)
                        self.assertIn("?  Help", command_bar)
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
                    self.assertEqual(format_clock(64.9), "1:04")
                    self.assertEqual(progress_bar(60.0, 180.0), "━━━━────────")
                    self.assertNotEqual(pulse_meter(0.5, 0), pulse_meter(0.5, 1))
                    self.assertEqual(queue_marker("now"), "▶")
                    self.assertEqual(queue_marker("+1"), "1")
                    self.assertEqual(confidence_label("medium"), "med")
                    fit = fit_cell("caution low-vocal", palette=PALETTE_BY_KEY["warmline"])
                    self.assertEqual(fit.plain, "caution low-vocal")
                    self.assertGreaterEqual(len(fit.spans), 2)
                    self.assertIsNotNone(app.store)
                    track = app.store.get_track(track_id) if app.store is not None else None
                    self.assertIsNotNone(track)
                    candidate = CandidateScore(
                        track=track,
                        phase=SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid"),
                        score=1.0,
                        confidence="high",
                        reasons=("semantic risk: dramatic for low-stimulation phase", "vocalness feature supports no-vocals constraint"),
                    )
                    self.assertEqual(app.fit_label(candidate), "caution low-vocal")
                    await pilot.press("q")

    async def test_tui_play_starts_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
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
                self.add_ready_track(store, tmp, "a.mp3")
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
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
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

    async def test_tui_next_and_previous_do_not_record_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                self.add_ready_track(store, tmp, "c.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    self.assertIsNotNone(app.runner)
                    before = app.runner.current() if app.runner is not None else None
                    await pilot.press(">")
                    after_next = app.runner.current() if app.runner is not None else None
                    await pilot.press("<")
                    after_previous = app.runner.current() if app.runner is not None else None
                    await pilot.press("q")

                store = TonepathStore()
                feedback_rows = store.conn.execute("SELECT COUNT(*) AS count FROM feedback").fetchone()["count"]
                store.close()
                self.assertIsNotNone(before)
                self.assertIsNotNone(after_next)
                self.assertIsNotNone(after_previous)
                self.assertNotEqual(before.track.id, after_next.track.id)
                self.assertEqual(before.track.id, after_previous.track.id)
                self.assertEqual(feedback_rows, 0)

    async def test_tui_next_replaces_playback_without_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()) as start, patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press(">")
                        await pilot.press("q")

                store = TonepathStore()
                feedback_rows = store.conn.execute("SELECT COUNT(*) AS count FROM feedback").fetchone()["count"]
                store.close()
                self.assertGreaterEqual(start.call_count, 2)
                self.assertTrue(stop.called)
                self.assertEqual(feedback_rows, 0)

    async def test_tui_playback_mode_cycles_and_help_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    self.assertEqual(app.playback_mode, "Manual")
                    await pilot.press("m")
                    self.assertEqual(app.playback_mode, "Continue Path")
                    await pilot.press("m")
                    self.assertEqual(app.playback_mode, "Repeat One")
                    await pilot.press("?")
                    self.assertEqual(app.right_panel, "help")
                    self.assertIn("Playback", app.why_panel_text())
                    self.assertIn("Feedback", app.why_panel_text())
                    self.assertIn("Tools", app.why_panel_text())
                    self.assertIn("Theme", app.why_panel_text())
                    self.assertIn("cycle Warmline", app.why_panel_text())
                    self.assertIn("Solarized", app.why_panel_text())
                    self.assertIn("Catppuccin", app.why_panel_text())
                    self.assertIn("Dracula", app.why_panel_text())
                    self.assertIn("Jukebox", app.why_panel_text())
                    self.assertIn("next track, no feedback", app.why_panel_text())
                    self.assertIn("finish prompt editing", app.why_panel_text())
                    self.assertIn("quit when prompt is not focused", app.why_panel_text())
                    self.assertIn("quit anytime", app.why_panel_text())
                    with patch.object(app, "log_event") as log_event:
                        await pilot.press("e")
                        self.assertTrue(app.events_expanded)
                    log_event.assert_not_called()
                    await pilot.press("q")

    async def test_tui_ai_assist_panel_is_read_only_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(
                os.environ,
                {"TONEPATH_HOME": str(home), "TONEPATH_LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "secret-token"},
                clear=True,
            ):
                config.write_config(config.preset_config("smart", send_to_llm=True))
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()
                before = config.render_config(config.load_config())

                app = TonepathApp()
                async with app.run_test() as pilot:
                    app.query_one("#prompt-input", Input).blur()
                    await pilot.press("i")
                    panel = app.why_panel_text()
                    self.assertEqual(app.right_panel, "ai_assist")
                    self.assertIn("AI Assist", panel)
                    self.assertIn("Status: AI Assist Ready: deepseek", panel)
                    self.assertIn("Will call LLM: yes, on new prompts", panel)
                    self.assertNotIn("secret-token", panel)
                    self.assertIn("AI deepseek", app.status_bar_text())
                    await pilot.press("q")

                after = config.render_config(config.load_config())
                self.assertEqual(before, after)

    async def test_tui_continue_path_starts_next_track_on_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FinishedProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("m")
                        self.assertEqual(app.playback_mode, "Continue Path")
                        await pilot.press("space")
                        app.poll_playback_finished()
                        self.assertEqual(app.playback_status, "Playing")
                        self.assertEqual(app.runner.current_index if app.runner is not None else -1, 1)
                        await pilot.press("q")
                self.assertGreaterEqual(start.call_count, 2)

    async def test_tui_command_bar_stays_visible_when_prompt_is_focused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    command_bar = app.query_one("#command-bar").render().plain
                    self.assertIn("Enter", command_bar)
                    self.assertIn("Submit", command_bar)
                    self.assertIn("Space", command_bar)
                    self.assertIn("Play", command_bar)
                    self.assertIn(">", command_bar)
                    self.assertIn("Next", command_bar)
                    self.assertIn("s", command_bar)
                    self.assertIn("Skip", command_bar)
                    self.assertIn("Esc", command_bar)
                    self.assertIn("Done", command_bar)
                    self.assertIn("Ctrl+Q", command_bar)
                    self.assertIn("Quit", command_bar)
                    await pilot.press("ctrl+q")

    async def test_tui_q_remains_prompt_text_when_prompt_is_focused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt_input = app.query_one("#prompt-input", Input)
                    self.assertTrue(prompt_input.has_focus)
                    await pilot.press("q")
                    self.assertEqual(prompt_input.value, "q")
                    self.assertIsNone(app.runner)
                    await pilot.press("ctrl+q")

    async def test_tui_escape_blurs_prompt_without_submitting_or_clearing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp()
                async with app.run_test() as pilot:
                    prompt_input = app.query_one("#prompt-input", Input)
                    prompt_input.value = "quiet focus"
                    self.assertTrue(prompt_input.has_focus)
                    await pilot.press("escape")
                    self.assertFalse(prompt_input.has_focus)
                    self.assertEqual(prompt_input.value, "quiet focus")
                    self.assertIsNone(app.runner)
                    command_bar = app.query_one("#command-bar").render().plain
                    self.assertNotIn("Submit", command_bar)
                    self.assertIn("Space", command_bar)
                    self.assertIn("Help", command_bar)
                    await pilot.press("q")

    async def test_tui_ctrl_q_quits_from_focused_prompt_and_stops_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FakeProcess()), patch.object(
                    MpvAdapter, "stop_process"
                ) as stop:
                    async with app.run_test() as pilot:
                        await pilot.press("space")
                        await pilot.press("/")
                        self.assertTrue(app.query_one("#prompt-input", Input).has_focus)
                        await pilot.press("ctrl+q")
                self.assertTrue(stop.called)

    async def test_tui_theme_cycles_and_persists_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                config.write_config(config.default_config())
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    self.assertEqual(app.theme_key, "warmline")
                    await pilot.press("t")
                    self.assertEqual(app.theme_key, "midnight")
                    self.assertEqual(config.load_config().ui.theme, "midnight")
                    self.assertEqual(app.palette.label, "Midnight")
                    await pilot.press("t")
                    self.assertEqual(app.theme_key, "high-contrast")
                    self.assertEqual(config.load_config().ui.theme, "high-contrast")
                    await pilot.press("q")

    async def test_tui_progress_text_uses_local_playback_clock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                track_id = self.add_ready_track(store, tmp, "a.mp3")
                store.conn.execute("UPDATE tracks SET duration = ? WHERE id = ?", (180.0, track_id))
                store.conn.commit()
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    self.assertIn("--:--", app.progress_text(None))
                    app.playback_status = "Playing"
                    app.playback_started_at = time.monotonic() - 64.0
                    self.assertIn("1:04", app.progress_text(180.0))
                    self.assertIn("3:00", app.progress_text(180.0))
                    self.assertIn("━", app.progress_text(180.0))
                    await pilot.press("q")

    async def test_tui_repeat_one_restarts_same_track_on_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FinishedProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("m")
                        await pilot.press("m")
                        self.assertEqual(app.playback_mode, "Repeat One")
                        await pilot.press("space")
                        app.poll_playback_finished()
                        self.assertEqual(app.playback_status, "Playing")
                        self.assertEqual(app.runner.current_index if app.runner is not None else -1, 0)
                        await pilot.press("q")
                self.assertGreaterEqual(start.call_count, 2)

    async def test_tui_repeat_path_wraps_to_first_track_on_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
                self.add_ready_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                with patch.object(MpvAdapter, "start", return_value=FinishedProcess()) as start:
                    async with app.run_test() as pilot:
                        await pilot.press("m")
                        await pilot.press("m")
                        await pilot.press("m")
                        self.assertEqual(app.playback_mode, "Repeat Path")
                        if app.runner is not None:
                            app.runner.current_index = len(app.runner.queue) - 1
                        await pilot.press("space")
                        app.poll_playback_finished()
                        self.assertEqual(app.playback_status, "Playing")
                        self.assertEqual(app.runner.current_index if app.runner is not None else -1, 0)
                        await pilot.press("q")
                self.assertGreaterEqual(start.call_count, 2)

    async def test_tui_stop_key_stops_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_ready_track(store, tmp, "a.mp3")
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
                self.add_ready_track(store, tmp, "a.mp3")
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
        self.configure_music_dir(tmp)
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

    def add_ready_track(self, store: TonepathStore, tmp: str, name: str) -> int:
        track_id = self.add_track(store, tmp, name)
        store.upsert_features(
            TrackFeatures(
                track_id=track_id,
                bpm=100.0,
                loudness=-14.0,
                energy=0.5,
                vocalness=0.2,
                feature_source="test",
                confidence="high",
            )
        )
        return track_id

    def configure_music_dir(self, tmp: str) -> None:
        current = config.load_config()
        config.write_config(
            config.TonepathConfig(
                music_dirs=(str(Path(tmp)),),
                data_dir=current.data_dir,
                player=current.player,
                network_mode=current.network_mode,
                privacy=current.privacy,
                models=current.models,
                experience=current.experience,
            )
        )


if __name__ == "__main__":
    unittest.main()
