import tempfile
import unittest
from pathlib import Path

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
