import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import SessionPhase, SessionPlan, SessionRequest, Track
from tonepath.playback_controller import CURRENT_PLAY_ID_KEY, PlaybackController
from tonepath.selector import empty_path_guidance, score_track, select_path


runner = CliRunner()


class FakeProcess:
    pid = 4321


class TrackReactionPersistenceTest(unittest.TestCase):
    def test_reaction_is_stable_and_clear_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tonepath.db"
            store = TonepathStore(path)
            track_id = add_track(store, tmp, "a.mp3")

            store.set_track_reaction(track_id, "liked")
            store.set_track_reaction(track_id, "disliked")
            self.assertEqual(store.get_track_reaction(track_id), "disliked")

            store.clear_track_reaction(track_id)
            store.clear_track_reaction(track_id)
            self.assertIsNone(store.get_track_reaction(track_id))
            store.close()

    def test_legacy_track_like_is_migrated_once_and_does_not_revive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tonepath.db"
            store = TonepathStore(path)
            track_id = add_track(store, tmp, "legacy.mp3")
            store.record_feedback("like", track_id=track_id)
            store.conn.execute("DROP TABLE track_reactions")
            store.conn.execute("DELETE FROM app_state WHERE key = 'track_reactions_v1_migrated'")
            store.conn.commit()
            store.close()

            migrated = TonepathStore(path)
            self.assertEqual(migrated.get_track_reaction(track_id), "liked")
            migrated.clear_track_reaction(track_id)
            migrated.close()

            reopened = TonepathStore(path)
            self.assertIsNone(reopened.get_track_reaction(track_id))
            reopened.close()

    def test_track_deletion_cascades_reaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = add_track(store, tmp, "a.mp3")
            store.set_track_reaction(track_id, "liked")

            store.conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            store.conn.commit()

            self.assertEqual(store.list_track_reactions(), [])
            store.close()

    def test_reaction_listing_uses_effective_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            liked_id = add_track(store, tmp, "liked.mp3", title="Liked", artist="Artist")
            disliked_id = add_track(store, tmp, "disliked.mp3", title="Disliked", artist="Other")
            store.upsert_metadata_override(disliked_id, "metadata:title", "Restored Title")
            store.upsert_metadata_override(disliked_id, "metadata:artist", "Restored Artist")
            store.set_track_reaction(liked_id, "liked")
            store.set_track_reaction(disliked_id, "disliked")

            rows = store.list_track_reactions("disliked")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["track_id"], disliked_id)
            self.assertEqual(rows[0]["reaction"], "disliked")
            self.assertEqual(rows[0]["title"], "Restored Title")
            self.assertEqual(rows[0]["artist"], "Restored Artist")
            store.close()


class TrackReactionSelectorTest(unittest.TestCase):
    def test_like_adds_one_fixed_bonus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = add_track(store, tmp, "liked.mp3")
            track = store.get_track(track_id)
            self.assertIsNotNone(track)
            phase = SessionPhase("focus", 0, 600, 0.5, 0.55, 0.5)
            baseline = score_track(store, track, phase)

            store.set_track_reaction(track_id, "liked")
            liked = score_track(store, track, phase)
            store.set_track_reaction(track_id, "liked")
            liked_again = score_track(store, track, phase)

            self.assertEqual(liked.score - baseline.score, 1.5)
            self.assertEqual(liked_again.score, liked.score)
            self.assertIn("you liked this track", liked.reasons)
            store.close()

    def test_disliked_track_is_excluded_from_new_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            hidden_id = add_track(store, tmp, "hidden.mp3")
            visible_id = add_track(store, tmp, "visible.mp3")
            store.set_track_reaction(hidden_id, "disliked")

            selected = select_path(store, plan(), limit_per_phase=2)

            self.assertEqual([candidate.track.id for candidate in selected], [visible_id])
            store.close()

    def test_all_disliked_tracks_produce_an_empty_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = add_track(store, tmp, "hidden.mp3")
            store.set_track_reaction(track_id, "disliked")

            self.assertEqual(select_path(store, plan(), limit_per_phase=2), [])
            guidance = empty_path_guidance(store)
            self.assertIn("All scanned tracks are Disliked", guidance)
            self.assertIn("feedback clear TRACK_ID", guidance)
            store.close()


class TrackReactionCliTest(unittest.TestCase):
    def test_cli_sets_lists_and_clears_explicit_track_reaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = TonepathStore()
                track_id = add_track(store, tmp, "a.mp3", title="A", artist="Artist")
                store.close()

                liked = runner.invoke(app, ["feedback", "like", str(track_id)])
                listed = runner.invoke(app, ["feedback", "reactions", "--state", "liked"])
                cleared = runner.invoke(app, ["feedback", "clear", str(track_id)])

                self.assertEqual(liked.exit_code, 0, liked.output)
                self.assertIn("Liked", liked.output)
                self.assertEqual(listed.exit_code, 0, listed.output)
                self.assertIn(str(track_id), listed.output)
                self.assertIn("A", listed.output)
                self.assertEqual(cleared.exit_code, 0, cleared.output)
                self.assertIn("Reaction cleared", cleared.output)
                check = TonepathStore()
                self.assertIsNone(check.get_track_reaction(track_id))
                check.close()

    def test_cli_without_track_id_uses_active_playback_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = TonepathStore()
                track_id = add_track(store, tmp, "a.mp3")
                play_id = store.start_play(session_id=None, track_id=track_id)
                store.set_app_state(CURRENT_PLAY_ID_KEY, str(play_id))
                store.close()

                result = runner.invoke(app, ["feedback", "dislike"])

                self.assertEqual(result.exit_code, 0, result.output)
                check = TonepathStore()
                self.assertEqual(check.get_track_reaction(track_id), "disliked")
                check.close()

    def test_cli_without_track_id_or_active_playback_is_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                TonepathStore().close()

                result = runner.invoke(app, ["feedback", "dislike"])

                self.assertEqual(result.exit_code, 2)
                self.assertIn("No active track", result.output)
                self.assertNotIn("Traceback", result.output)

    def test_cli_preview_plays_one_track_without_creating_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}, clear=True):
                store = TonepathStore()
                track_id = add_track(store, tmp, "a.mp3")
                store.close()

                with patch.object(PlaybackController, "replace", return_value=FakeProcess()) as replace:
                    result = runner.invoke(app, ["feedback", "play", str(track_id), "--background"])

                self.assertEqual(result.exit_code, 0, result.output)
                replace.assert_called_once()
                args, kwargs = replace.call_args
                self.assertEqual(args[0], [Path(tmp) / "a.mp3"])
                self.assertIsNone(kwargs["session_id"])
                self.assertEqual(kwargs["track_id"], track_id)
                check = TonepathStore()
                self.assertEqual(check.profile_summary()["sessions"], 0)
                check.close()

def add_track(
    store: TonepathStore,
    tmp: str,
    name: str,
    *,
    title: str | None = None,
    artist: str | None = "Artist",
) -> int:
    path = Path(tmp) / name
    path.write_bytes(b"audio")
    return store.upsert_track(
        Track(
            id=None,
            path=path,
            file_hash=name,
            mtime=1.0,
            title=title or name,
            artist=artist,
            album=None,
            genre=None,
            duration=120.0,
            format="mp3",
        )
    )


def plan() -> SessionPlan:
    request = SessionRequest("steady focus", "irritated", "focus", 1200)
    return SessionPlan(request, (SessionPhase("focus", 0, 1200, 0.5, 0.55, 0.5),))


if __name__ == "__main__":
    unittest.main()
