import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.history import (
    create_replay_session,
    export_history_bundle,
    list_history,
    load_history,
    prepare_replay,
)
from tonepath.models import CandidateScore, SessionPhase, SessionPlan, SessionRequest, Track
from tonepath.playback import MpvAdapter


class FakeProcess:
    pid = 4321


class HistoryPersistenceTest(unittest.TestCase):
    def test_new_history_tables_are_created_for_an_existing_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tonepath.db"
            store = TonepathStore(path)
            store.close()

            reopened = TonepathStore(path)
            tables = {
                str(row["name"])
                for row in reopened.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            self.assertIn("session_queue_items", tables)
            self.assertIn("session_bookmarks", tables)
            reopened.close()

    def test_replace_session_queue_preserves_order_and_display_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp, "a.mp3", title="Original", artist="Artist")
            plan = self.plan()
            session_id = store.save_session(plan)
            track = store.get_track(track_id)
            self.assertIsNotNone(track)
            candidate = CandidateScore(
                track=track,
                phase=plan.phases[0],
                score=1.25,
                confidence="high",
                reasons=("low vocalness", "phase energy fit"),
            )

            store.replace_session_queue(session_id, [candidate])
            rows = store.session_queue_items(session_id)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["position"], 0)
            self.assertEqual(rows[0]["track_id"], track_id)
            self.assertEqual(rows[0]["track_path"], str(track.path))
            self.assertEqual(rows[0]["title"], "Original")
            self.assertEqual(rows[0]["artist"], "Artist")
            self.assertEqual(rows[0]["phase_label"], "focus")
            self.assertEqual(rows[0]["score"], 1.25)
            self.assertEqual(rows[0]["confidence"], "high")
            self.assertEqual(rows[0]["reasons"], ["low vocalness", "phase energy fit"])
            store.close()

    def test_track_deletion_keeps_queue_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp, "a.mp3", title="Snapshot", artist="Artist")
            plan = self.plan()
            session_id = store.save_session(plan)
            track = store.get_track(track_id)
            self.assertIsNotNone(track)
            store.replace_session_queue(
                session_id,
                [CandidateScore(track, plan.phases[0], 1.0, "high", ("selected",))],
            )

            store.conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            store.conn.commit()
            rows = store.session_queue_items(session_id)

            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["track_id"])
            self.assertEqual(rows[0]["title"], "Snapshot")
            self.assertEqual(rows[0]["artist"], "Artist")
            store.close()

    def test_session_deletion_cascades_queue_and_bookmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp, "a.mp3")
            plan = self.plan()
            session_id = store.save_session(plan)
            track = store.get_track(track_id)
            self.assertIsNotNone(track)
            store.replace_session_queue(
                session_id,
                [CandidateScore(track, plan.phases[0], 1.0, "high", ("selected",))],
            )
            store.save_session_bookmark(session_id, "Focus path")

            store.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            store.conn.commit()

            queue_count = store.conn.execute(
                "SELECT COUNT(*) AS count FROM session_queue_items"
            ).fetchone()["count"]
            bookmark_count = store.conn.execute(
                "SELECT COUNT(*) AS count FROM session_bookmarks"
            ).fetchone()["count"]
            self.assertEqual(queue_count, 0)
            self.assertEqual(bookmark_count, 0)
            store.close()

    def test_profile_delete_reports_and_clears_history_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = self.add_track(store, tmp, "a.mp3")
            plan = self.plan()
            session_id = store.save_session(plan)
            track = store.get_track(track_id)
            self.assertIsNotNone(track)
            store.replace_session_queue(
                session_id,
                [CandidateScore(track, plan.phases[0], 1.0, "high", ("selected",))],
            )
            store.save_session_bookmark(session_id, "Focus path")

            before = store.profile_summary()
            self.assertEqual(before["session_queue_items"], 1)
            self.assertEqual(before["session_bookmarks"], 1)

            store.delete_profile_data()

            after = store.profile_summary()
            self.assertEqual(after["sessions"], 0)
            self.assertEqual(after["session_queue_items"], 0)
            self.assertEqual(after["session_bookmarks"], 0)
            self.assertEqual(after["tracks"], 1)
            store.close()

    @staticmethod
    def plan() -> SessionPlan:
        request = SessionRequest("focus", "irritated", "focus", 1800, no_vocals=True)
        phase = SessionPhase("focus", 0, 1800, 0.35, 0.55, 0.35, "avoid")
        return SessionPlan(request, (phase,))

    @staticmethod
    def add_track(
        store: TonepathStore,
        tmp: str,
        name: str,
        title: str | None = None,
        artist: str | None = None,
    ) -> int:
        path = Path(tmp) / name
        path.write_bytes(b"not real audio")
        return store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash=name,
                mtime=1.0,
                title=title or name,
                artist=artist or "artist",
                album=None,
                genre=None,
                duration=120.0,
                format="mp3",
            )
        )


class HistoryDomainTest(unittest.TestCase):
    def test_default_history_only_includes_played_or_saved_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(store, tmp, "a.mp3")
            hidden_id = store.save_session(HistoryPersistenceTest.plan())
            played_id = store.save_session(HistoryPersistenceTest.plan())
            saved_id = store.save_session(HistoryPersistenceTest.plan())
            store.start_play(played_id, track_id)
            store.save_session_bookmark(saved_id, "Saved")

            default_ids = [session.id for session in list_history(store)]
            all_ids = [session.id for session in list_history(store, include_all=True)]
            saved_ids = [session.id for session in list_history(store, saved_only=True)]

            self.assertNotIn(hidden_id, default_ids)
            self.assertEqual(set(default_ids), {played_id, saved_id})
            self.assertEqual(set(all_ids), {hidden_id, played_id, saved_id})
            self.assertEqual(saved_ids, [saved_id])
            store.close()

    def test_load_history_keeps_snapshot_after_track_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(
                store,
                tmp,
                "a.mp3",
                title="Historical title",
                artist="Historical artist",
            )
            session_id = self.save_queue(store, [track_id])
            store.conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            store.conn.commit()

            record = load_history(store, session_id)

            self.assertEqual(record.queue[0].title, "Historical title")
            self.assertEqual(record.queue[0].artist, "Historical artist")
            self.assertIsNone(record.queue[0].track_id)
            store.close()

    def test_prepare_replay_preserves_order_and_omits_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            first_id = HistoryPersistenceTest.add_track(store, tmp, "first.mp3")
            missing_id = HistoryPersistenceTest.add_track(store, tmp, "missing.mp3")
            last_id = HistoryPersistenceTest.add_track(store, tmp, "last.mp3")
            source_session_id = self.save_queue(store, [first_id, missing_id, last_id])
            (Path(tmp) / "missing.mp3").rename(Path(tmp) / "moved-missing.mp3")

            replay = prepare_replay(store, source_session_id)

            self.assertEqual(
                [candidate.track.path.name for candidate in replay.candidates],
                ["first.mp3", "last.mp3"],
            )
            self.assertEqual([item.path.name for item in replay.omitted], ["missing.mp3"])
            new_session_id = create_replay_session(store, replay)
            self.assertNotEqual(new_session_id, source_session_id)
            self.assertEqual(
                [Path(str(row["track_path"])).name for row in store.session_queue_items(new_session_id)],
                ["first.mp3", "last.mp3"],
            )
            self.assertEqual(len(store.session_queue_items(source_session_id)), 3)
            store.close()

    def test_prepare_replay_rejects_legacy_session_without_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            session_id = store.save_session(HistoryPersistenceTest.plan())

            with self.assertRaisesRegex(RuntimeError, "does not have a saved queue snapshot"):
                prepare_replay(store, session_id)
            store.close()

    def test_prepare_replay_rejects_when_every_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(store, tmp, "missing.mp3")
            session_id = self.save_queue(store, [track_id])
            (Path(tmp) / "missing.mp3").rename(Path(tmp) / "moved-missing.mp3")

            with self.assertRaisesRegex(RuntimeError, "No playable tracks remain"):
                prepare_replay(store, session_id)
            store.close()

    def test_export_writes_json_and_utf8_playlist_without_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            first_id = HistoryPersistenceTest.add_track(
                store,
                tmp,
                "一首歌.mp3",
                title="一首歌",
                artist="Artist",
            )
            missing_id = HistoryPersistenceTest.add_track(store, tmp, "missing.mp3")
            session_id = self.save_queue(store, [first_id, missing_id])
            store.record_feedback("like", session_id=session_id, track_id=first_id)
            (Path(tmp) / "missing.mp3").rename(Path(tmp) / "moved-missing.mp3")
            output = Path(tmp) / "export"

            export_history_bundle(store, session_id, output)

            payload = (output / "session.json").read_text(encoding="utf-8")
            playlist = (output / "playlist.m3u8").read_text(encoding="utf-8")
            self.assertIn('"prompt": "focus"', payload)
            self.assertIn('"type": "like"', payload)
            self.assertIn('"omitted"', payload)
            self.assertIn('"contains_local_paths": true', payload)
            self.assertIn("一首歌.mp3", playlist)
            self.assertNotIn("missing.mp3", playlist)
            store.close()

    def test_export_refuses_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(store, tmp, "a.mp3")
            session_id = self.save_queue(store, [track_id])
            output = Path(tmp) / "export"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                export_history_bundle(store, session_id, output)
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")
            store.close()

    def test_export_marks_track_available_when_current_path_was_renamed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(store, tmp, "old.mp3")
            session_id = self.save_queue(store, [track_id])
            new_path = Path(tmp) / "new.mp3"
            (Path(tmp) / "old.mp3").rename(new_path)
            store.conn.execute(
                "UPDATE tracks SET path = ? WHERE id = ?",
                (str(new_path), track_id),
            )
            store.conn.commit()
            output = Path(tmp) / "export"

            export_history_bundle(store, session_id, output)

            payload = json.loads((output / "session.json").read_text(encoding="utf-8"))
            playlist = (output / "playlist.m3u8").read_text(encoding="utf-8")
            self.assertTrue(payload["queue"][0]["available"])
            self.assertEqual(payload["omitted"], [])
            self.assertIn(str(new_path), playlist)
            store.close()

    def test_export_sanitizes_m3u_metadata_line_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(
                store,
                tmp,
                "a.mp3",
                title="Line\nBreak",
                artist="Artist\rName",
            )
            session_id = self.save_queue(store, [track_id])
            output = Path(tmp) / "export"

            export_history_bundle(store, session_id, output)

            playlist = (output / "playlist.m3u8").read_text(encoding="utf-8")
            self.assertIn("#EXTINF:-1,Artist Name - Line Break", playlist)
            self.assertEqual(len(playlist.splitlines()), 3)
            store.close()

    def test_export_rejects_playlist_path_with_line_break(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = HistoryPersistenceTest.add_track(store, tmp, "bad\npath.mp3")
            session_id = self.save_queue(store, [track_id])
            output = Path(tmp) / "export"

            with self.assertRaisesRegex(RuntimeError, "line break"):
                export_history_bundle(store, session_id, output)
            self.assertFalse((output / "session.json").exists())
            self.assertFalse((output / "playlist.m3u8").exists())
            store.close()

    @staticmethod
    def save_queue(store: TonepathStore, track_ids: list[int]) -> int:
        plan = HistoryPersistenceTest.plan()
        session_id = store.save_session(plan)
        candidates: list[CandidateScore] = []
        for position, track_id in enumerate(track_ids):
            track = store.get_track(track_id)
            if track is None:
                raise AssertionError("test track missing")
            candidates.append(
                CandidateScore(
                    track,
                    plan.phases[0],
                    1.0 - position / 10,
                    "high",
                    (f"reason {position}",),
                )
            )
        store.replace_session_queue(session_id, candidates)
        return session_id


class HistoryCliTest(unittest.TestCase):
    def test_list_filters_and_save_unsave_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                track_id = HistoryPersistenceTest.add_track(store, tmp, "a.mp3")
                hidden_id = store.save_session(HistoryPersistenceTest.plan())
                played_id = HistoryDomainTest.save_queue(store, [track_id])
                store.start_play(played_id, track_id)
                store.close()

                default_result = CliRunner().invoke(app, ["history", "list"])
                all_result = CliRunner().invoke(app, ["history", "list", "--all"])
                save_result = CliRunner().invoke(
                    app,
                    ["history", "save", str(hidden_id), "--name", "写代码"],
                )
                save_again = CliRunner().invoke(
                    app,
                    ["history", "save", str(hidden_id), "--name", "写代码"],
                )
                saved_result = CliRunner().invoke(app, ["history", "list", "--saved-only"])
                unsave_result = CliRunner().invoke(app, ["history", "unsave", str(hidden_id)])
                unsave_again = CliRunner().invoke(app, ["history", "unsave", str(hidden_id)])

                self.assertEqual(default_result.exit_code, 0, default_result.output)
                self.assertIn(f"\n   {played_id} ", default_result.output)
                self.assertNotIn(f"\n   {hidden_id} ", default_result.output)
                self.assertEqual(all_result.exit_code, 0, all_result.output)
                self.assertIn(f"\n   {hidden_id} ", all_result.output)
                self.assertEqual(save_result.exit_code, 0, save_result.output)
                self.assertEqual(save_again.exit_code, 0, save_again.output)
                self.assertIn("写代码", saved_result.output)
                self.assertEqual(unsave_result.exit_code, 0, unsave_result.output)
                self.assertEqual(unsave_again.exit_code, 0, unsave_again.output)

    def test_show_includes_original_request_and_rebuild_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                session_id = store.save_session(HistoryPersistenceTest.plan())
                store.close()

                result = CliRunner().invoke(app, ["history", "show", str(session_id)])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Request: focus", result.output)
                self.assertIn("uv run tonepath listen focus", result.output)
                self.assertIn("does not have a queue snapshot", result.output)

    def test_history_output_displays_rich_markup_as_literal_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                track_id = HistoryPersistenceTest.add_track(
                    store,
                    tmp,
                    "a.mp3",
                    title="[green]Track[/green]",
                    artist="[blue]Artist[/blue]",
                )
                plan = HistoryPersistenceTest.plan()
                session_id = store.save_session(
                    SessionPlan(
                        request=SessionRequest("[bold]focus[/bold]", "irritated", "focus", 1800),
                        phases=plan.phases,
                    )
                )
                track = store.get_track(track_id)
                self.assertIsNotNone(track)
                store.replace_session_queue(
                    session_id,
                    [CandidateScore(track, plan.phases[0], 1.0, "high", ("selected",))],
                )
                store.save_session_bookmark(session_id, "[red]Saved[/red]")
                store.close()

                list_result = CliRunner().invoke(app, ["history", "list", "--saved-only"])
                show_result = CliRunner().invoke(app, ["history", "show", str(session_id)])

                self.assertEqual(list_result.exit_code, 0, list_result.output)
                self.assertEqual(show_result.exit_code, 0, show_result.output)
                self.assertIn("[red]Save", list_result.output)
                self.assertIn("[bold]focus[/bold]", show_result.output)
                self.assertIn("[green]Track[/green]", show_result.output)

    def test_replay_dry_run_preserves_order_without_writing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                first_id = HistoryPersistenceTest.add_track(store, tmp, "first.mp3")
                second_id = HistoryPersistenceTest.add_track(store, tmp, "second.mp3")
                session_id = HistoryDomainTest.save_queue(store, [first_id, second_id])
                before_count = store.conn.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()["count"]
                store.close()

                result = CliRunner().invoke(
                    app,
                    ["history", "replay", str(session_id), "--dry-run"],
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertLess(result.output.index("first.mp3"), result.output.index("second.mp3"))
                self.assertIn("Dry-run only; replay session not saved.", result.output)
                reopened = TonepathStore()
                after_count = reopened.conn.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()["count"]
                self.assertEqual(after_count, before_count)
                reopened.close()

    def test_background_replay_creates_new_session_and_records_first_play(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                first_id = HistoryPersistenceTest.add_track(store, tmp, "first.mp3")
                second_id = HistoryPersistenceTest.add_track(store, tmp, "second.mp3")
                source_id = HistoryDomainTest.save_queue(store, [first_id, second_id])
                store.close()

                with patch.object(MpvAdapter, "start", return_value=FakeProcess()):
                    result = CliRunner().invoke(
                        app,
                        ["history", "replay", str(source_id), "--background"],
                    )

                self.assertEqual(result.exit_code, 0, result.output)
                reopened = TonepathStore()
                current_id = reopened.current_session_id()
                self.assertIsNotNone(current_id)
                self.assertNotEqual(current_id, source_id)
                queue = reopened.session_queue_items(current_id)
                self.assertEqual([row["track_id"] for row in queue], [first_id, second_id])
                play = reopened.conn.execute(
                    "SELECT session_id, track_id FROM plays ORDER BY id DESC LIMIT 1"
                ).fetchone()
                self.assertEqual(play["session_id"], current_id)
                self.assertEqual(play["track_id"], first_id)
                reopened.close()

    def test_export_command_writes_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            output = Path(tmp) / "bundle"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                store = TonepathStore()
                track_id = HistoryPersistenceTest.add_track(store, tmp, "a.mp3")
                session_id = HistoryDomainTest.save_queue(store, [track_id])
                store.close()

                result = CliRunner().invoke(
                    app,
                    ["history", "export", str(session_id), "--output", str(output)],
                )

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue((output / "session.json").is_file())
                self.assertTrue((output / "playlist.m3u8").is_file())
                self.assertIn("local file paths", result.output)

    def test_export_command_reports_data_and_filesystem_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                for error in (ValueError("bad queue data"), OSError("disk full")):
                    with self.subTest(error=type(error).__name__), patch(
                        "tonepath.cli.export_history_bundle",
                        side_effect=error,
                    ):
                        result = CliRunner().invoke(
                            app,
                            ["history", "export", "1", "--output", str(Path(tmp) / "bundle")],
                        )

                    self.assertEqual(result.exit_code, 2, result.output)
                    self.assertIn(str(error), result.output)
                    self.assertNotIn("Traceback", result.output)


if __name__ == "__main__":
    unittest.main()
