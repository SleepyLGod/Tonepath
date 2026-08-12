import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.db import TonepathStore
from tonepath.models import Track
from tonepath.session import SessionRunner


class SessionRunnerTest(unittest.TestCase):
    def test_skip_moves_current_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            self.add_track(store, tmp, "b.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            before = runner.current()
            runner.apply_feedback("skip")
            after = runner.current()
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            self.assertNotEqual(before.track.id, after.track.id)
            snapshot = store.session_queue_items(runner.session_id)
            self.assertEqual(
                [row["track_id"] for row in snapshot],
                [candidate.track.id for candidate in runner.queue],
            )
            store.close()

    def test_navigation_moves_without_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            self.add_track(store, tmp, "b.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            before = runner.current()
            moved_next = runner.move_next()
            after = runner.current()
            moved_previous = runner.move_previous()
            back = runner.current()
            feedback_rows = store.conn.execute("SELECT COUNT(*) AS count FROM feedback").fetchone()["count"]

            self.assertTrue(moved_next)
            self.assertTrue(moved_previous)
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            self.assertIsNotNone(back)
            self.assertNotEqual(before.track.id, after.track.id)
            self.assertEqual(before.track.id, back.track.id)
            self.assertEqual(feedback_rows, 0)
            store.close()

    def test_initial_queue_is_saved_with_the_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            self.add_track(store, tmp, "b.mp3")

            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            snapshot = store.session_queue_items(runner.session_id)

            self.assertEqual(
                [row["track_id"] for row in snapshot],
                [candidate.track.id for candidate in runner.queue],
            )
            store.close()

    def test_no_vocals_updates_session_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            runner.apply_feedback("no-vocals")
            self.assertTrue(runner.active_plan().request.no_vocals)
            self.assertTrue(all(phase.vocal_policy == "avoid" for phase in runner.active_plan().phases))
            store.close()

    def test_too_loud_reduces_later_energy_preference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            before = runner.active_plan().phases[0].target_energy
            runner.apply_feedback("too-loud")
            after = runner.active_plan().phases[0].target_energy
            self.assertLess(after, before)
            store.close()

    def test_track_reaction_toggles_without_changing_current_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            self.add_track(store, tmp, "b.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            current = runner.current()
            self.assertIsNotNone(current)
            track_id = current.track.id
            original_queue = list(runner.queue)
            original_index = runner.current_index

            liked = runner.toggle_track_reaction("liked")
            disliked = runner.toggle_track_reaction("disliked")
            cleared = runner.toggle_track_reaction("disliked")

            self.assertEqual(liked, "Liked; future Requests will remember this track.")
            self.assertEqual(disliked, "Disliked; future Requests will hide this track.")
            self.assertEqual(cleared, "Reaction cleared.")
            self.assertIsNone(store.get_track_reaction(track_id))
            self.assertEqual(runner.queue, original_queue)
            self.assertEqual(runner.current_index, original_index)
            store.close()

    def test_legacy_like_feedback_uses_stable_reaction_without_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            self.add_track(store, tmp, "b.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            current = runner.current()
            self.assertIsNotNone(current)
            original_queue = list(runner.queue)
            original_index = runner.current_index

            message = runner.apply_feedback("like")

            self.assertEqual(message, "Liked; future Requests will remember this track.")
            self.assertEqual(store.get_track_reaction(current.track.id), "liked")
            self.assertEqual(runner.queue, original_queue)
            self.assertEqual(runner.current_index, original_index)
            feedback_count = store.conn.execute(
                "SELECT COUNT(*) AS count FROM feedback WHERE type = 'like'"
            ).fetchone()["count"]
            self.assertEqual(feedback_count, 0)
            store.close()

    def test_rebuild_failure_keeps_active_queue_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            self.add_track(store, tmp, "a.mp3")
            self.add_track(store, tmp, "b.mp3")
            runner = SessionRunner(store, "from irritated to focus in 30 minutes")
            original_queue = list(runner.queue)

            with patch("tonepath.session.select_path", return_value=[]), patch.object(
                store,
                "replace_session_queue",
                side_effect=RuntimeError("database write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "database write failed"):
                    runner.rebuild_future()

            self.assertEqual(runner.queue, original_queue)
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
