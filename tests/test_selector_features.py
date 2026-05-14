import tempfile
import unittest
from pathlib import Path

from tonepath.analysis import AUDIO_SEPARATOR_FEATURE_SOURCE, ESSENTIA_VOICE_FEATURE_SOURCE
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

    def test_bpm_scores_higher_for_higher_energy_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            steady_id = store.upsert_track(track_for(tmp, "steady.wav"))
            slow_id = store.upsert_track(track_for(tmp, "slow.wav"))
            store.upsert_features(
                TrackFeatures(
                    steady_id,
                    bpm=128.0,
                    loudness=-14.0,
                    energy=0.7,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    slow_id,
                    bpm=60.0,
                    loudness=-14.0,
                    energy=0.7,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )

            phase = SessionPhase("lift", 0, 600, 0.7, 0.6, 0.75)
            steady = score_track(store, store.get_track(steady_id), phase)
            slow = score_track(store, store.get_track(slow_id), phase)

            self.assertGreater(steady.score, slow.score)
            self.assertIn("BPM feature contributes to phase fit", steady.reasons)
            store.close()

    def test_high_bpm_is_not_preferred_for_low_energy_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            calm_id = store.upsert_track(track_for(tmp, "calm.wav"))
            frantic_id = store.upsert_track(track_for(tmp, "frantic.wav"))
            store.upsert_features(
                TrackFeatures(
                    calm_id,
                    bpm=82.0,
                    loudness=-28.0,
                    energy=0.25,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    frantic_id,
                    bpm=162.0,
                    loudness=-28.0,
                    energy=0.25,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )

            phase = SessionPhase("decompress", 0, 600, 0.35, 0.45, 0.25)
            calm = score_track(store, store.get_track(calm_id), phase)
            frantic = score_track(store, store.get_track(frantic_id), phase)

            self.assertGreater(calm.score, frantic.score)
            store.close()

    def test_low_vocalness_scores_higher_for_no_vocals_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            instrumental_id = store.upsert_track(track_for(tmp, "instrumental.wav"))
            vocal_id = store.upsert_track(track_for(tmp, "vocal.wav"))
            unknown_id = store.upsert_track(track_for(tmp, "unknown.wav"))
            store.upsert_features(
                TrackFeatures(
                    instrumental_id,
                    vocalness=0.2,
                    loudness=-20.0,
                    energy=0.4,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    vocal_id,
                    vocalness=0.8,
                    loudness=-20.0,
                    energy=0.4,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    unknown_id,
                    loudness=-20.0,
                    energy=0.4,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )

            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            instrumental = score_track(store, store.get_track(instrumental_id), phase)
            vocal = score_track(store, store.get_track(vocal_id), phase)
            unknown = score_track(store, store.get_track(unknown_id), phase)

            self.assertGreater(instrumental.score, unknown.score)
            self.assertGreater(unknown.score, vocal.score)
            self.assertIn("vocalness feature supports no-vocals constraint", instrumental.reasons)
            self.assertIn("vocalness feature conflicts with no-vocals constraint", vocal.reasons)
            self.assertIn("no-vocals requested but vocalness is unknown", unknown.reasons)
            store.close()

    def test_essentia_voice_source_is_weighted_above_separator_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            classifier_id = store.upsert_track(track_for(tmp, "classifier.wav"))
            separator_id = store.upsert_track(track_for(tmp, "separator.wav"))
            store.upsert_features(
                TrackFeatures(
                    classifier_id,
                    vocalness=0.2,
                    loudness=-20.0,
                    energy=0.4,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    separator_id,
                    vocalness=0.2,
                    loudness=-20.0,
                    energy=0.4,
                    feature_source=AUDIO_SEPARATOR_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            classifier = score_track(store, store.get_track(classifier_id), phase)
            separator = score_track(store, store.get_track(separator_id), phase)

            self.assertGreater(classifier.score, separator.score)
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
