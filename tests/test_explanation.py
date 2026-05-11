import tempfile
import unittest
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.explanation import explain_candidate
from tonepath.models import CandidateScore, SessionPhase, Track


class ExplanationTest(unittest.TestCase):
    def test_explanation_marks_unknown_bpm_without_inventing_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            audio = Path(tmp) / "测试歌曲.mp3"
            audio.write_bytes(b"not real audio")
            track = Track(
                id=None,
                path=audio,
                file_hash="hash",
                mtime=1.0,
                title="测试歌曲",
                artist="测试艺人",
                album=None,
                genre="ambient",
                duration=None,
                format="mp3",
            )
            track_id = store.upsert_track(track)
            persisted = store.get_track(track_id)
            self.assertIsNotNone(persisted)
            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            candidate = CandidateScore(persisted, phase, 1.0, "low", ("audio features unavailable",))
            explanation = explain_candidate(store, candidate)
            self.assertIn("BPM：unknown", explanation)
            self.assertNotIn("92 BPM", explanation)
            self.assertIn("Vocalness：unknown", explanation)
            store.close()


if __name__ == "__main__":
    unittest.main()
