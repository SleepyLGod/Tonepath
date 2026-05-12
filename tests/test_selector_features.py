import tempfile
import unittest
from pathlib import Path

from tonepath.db import TonepathStore
from tonepath.models import SessionPhase, Track, TrackFeatures
from tonepath.selector import score_track


class SelectorFeaturesTest(unittest.TestCase):
    def test_lower_energy_scores_higher_for_low_energy_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            quiet_id = store.upsert_track(track_for(tmp, "quiet.wav"))
            loud_id = store.upsert_track(track_for(tmp, "loud.wav"))
            store.upsert_features(
                TrackFeatures(quiet_id, loudness=-45.0, energy=0.1, feature_source="basic-local-analysis", confidence="medium")
            )
            store.upsert_features(
                TrackFeatures(loud_id, loudness=-6.0, energy=0.9, feature_source="basic-local-analysis", confidence="medium")
            )

            phase = SessionPhase("calm", 0, 600, 0.2, 0.5, 0.2)
            quiet = score_track(store, store.get_track(quiet_id), phase)
            loud = score_track(store, store.get_track(loud_id), phase)

            self.assertGreater(quiet.score, loud.score)
            self.assertEqual(quiet.confidence, "medium")
            self.assertIn("energy feature contributes to phase fit", quiet.reasons)
            self.assertIn("loudness feature contributes to phase fit", quiet.reasons)
            store.close()


def track_for(tmp: str, name: str) -> Track:
    path = Path(tmp) / name
    path.write_bytes(b"not real audio")
    return Track(
        id=None,
        path=path,
        file_hash=name,
        mtime=1.0,
        title=name,
        artist="artist",
        album=None,
        genre=None,
        duration=None,
        format="wav",
    )


if __name__ == "__main__":
    unittest.main()
