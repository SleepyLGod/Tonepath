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

    def test_focus_penalizes_low_vocalness_but_overstimulating_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            steady_id = store.upsert_track(track_for(tmp, "steady-instrumental.wav"))
            frantic_id = store.upsert_track(track_for(tmp, "frantic-instrumental.wav"))
            store.upsert_features(
                TrackFeatures(
                    steady_id,
                    bpm=96.0,
                    loudness=-16.0,
                    energy=0.46,
                    vocalness=0.18,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    frantic_id,
                    bpm=168.0,
                    loudness=-8.0,
                    energy=0.82,
                    vocalness=0.12,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            steady = score_track(store, store.get_track(steady_id), phase)
            frantic = score_track(store, store.get_track(frantic_id), phase)

            self.assertGreater(steady.score, frantic.score)
            self.assertIn("phase stimulation penalty adjusted the score", frantic.reasons)
            self.assertIn("low vocalness but overstimulating for this phase", frantic.reasons)
            store.close()

    def test_stabilize_penalizes_rush_e_style_no_vocals_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            steady_id = store.upsert_track(track_for(tmp, "steady-ost.wav"))
            rush_id = store.upsert_track(track_for(tmp, "rush-e.wav"))
            store.upsert_features(
                TrackFeatures(
                    steady_id,
                    bpm=96.0,
                    loudness=-11.36,
                    energy=0.621,
                    vocalness=0.185,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    rush_id,
                    bpm=143.5,
                    loudness=-13.32,
                    energy=0.556,
                    vocalness=0.141,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("stabilize", 0, 600, 0.45, 0.55, 0.45, "avoid")
            steady = score_track(store, store.get_track(steady_id), phase)
            rush = score_track(store, store.get_track(rush_id), phase)

            self.assertGreater(steady.score, rush.score)
            self.assertIn("low vocalness but overstimulating for this phase", rush.reasons)
            store.close()

    def test_calm_penalizes_rush_e_style_no_vocals_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            calm_id = store.upsert_track(track_for(tmp, "calm-instrumental.wav"))
            stable_inconclusive_id = store.upsert_track(track_for(tmp, "stable-inconclusive.wav"))
            rush_id = store.upsert_track(track_for(tmp, "rush-e.wav"))
            store.upsert_features(
                TrackFeatures(
                    calm_id,
                    bpm=92.0,
                    loudness=-16.5,
                    energy=0.38,
                    vocalness=0.18,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    stable_inconclusive_id,
                    bpm=108.0,
                    loudness=-13.0,
                    energy=0.52,
                    vocalness=0.52,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    rush_id,
                    bpm=143.5,
                    loudness=-13.32,
                    energy=0.556,
                    vocalness=0.141,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("calm", 0, 600, 0.14, 0.6, 0.14, "avoid")
            calm = score_track(store, store.get_track(calm_id), phase)
            stable_inconclusive = score_track(store, store.get_track(stable_inconclusive_id), phase)
            rush = score_track(store, store.get_track(rush_id), phase)

            self.assertGreater(calm.score, rush.score)
            self.assertGreater(stable_inconclusive.score, rush.score)
            self.assertIn("phase stimulation penalty adjusted the score", rush.reasons)
            self.assertIn("low vocalness but overstimulating for this phase", rush.reasons)
            store.close()

    def test_energized_phase_does_not_penalize_reasonable_high_bpm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            upbeat_id = store.upsert_track(track_for(tmp, "upbeat.wav"))
            slow_id = store.upsert_track(track_for(tmp, "slow.wav"))
            store.upsert_features(
                TrackFeatures(
                    upbeat_id,
                    bpm=143.0,
                    loudness=-10.0,
                    energy=0.72,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    slow_id,
                    bpm=78.0,
                    loudness=-16.0,
                    energy=0.35,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("energize", 0, 600, 0.7, 0.65, 0.75)
            upbeat = score_track(store, store.get_track(upbeat_id), phase)
            slow = score_track(store, store.get_track(slow_id), phase)

            self.assertGreater(upbeat.score, slow.score)
            self.assertNotIn("phase stimulation penalty adjusted the score", upbeat.reasons)
            store.close()

    def test_quiet_soften_penalizes_rush_e_style_track_without_no_vocals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav"))
            rush_id = store.upsert_track(track_for(tmp, "rush-e.wav"))
            store.upsert_features(
                TrackFeatures(
                    gentle_id,
                    bpm=92.0,
                    loudness=-16.5,
                    energy=0.38,
                    vocalness=0.35,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    rush_id,
                    bpm=143.6,
                    loudness=-21.63,
                    energy=0.279,
                    vocalness=0.115,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("soften", 0, 600, 0.22, 0.45, 0.22)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            rush = score_track(store, store.get_track(rush_id), phase)

            self.assertGreater(gentle.score, rush.score)
            self.assertIn("phase stimulation penalty adjusted the score", rush.reasons)
            store.close()

    def test_inconclusive_vocalness_does_not_outrank_low_vocalness_in_strict_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            low_vocal_id = store.upsert_track(track_for(tmp, "low-vocal.wav"))
            inconclusive_id = store.upsert_track(track_for(tmp, "inconclusive.wav"))
            store.upsert_features(
                TrackFeatures(
                    low_vocal_id,
                    bpm=92.0,
                    loudness=-16.0,
                    energy=0.45,
                    vocalness=0.22,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    inconclusive_id,
                    bpm=92.0,
                    loudness=-16.0,
                    energy=0.45,
                    vocalness=0.52,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            low_vocal = score_track(store, store.get_track(low_vocal_id), phase)
            inconclusive = score_track(store, store.get_track(inconclusive_id), phase)

            self.assertGreater(low_vocal.score, inconclusive.score)
            self.assertIn("vocalness feature is inconclusive for no-vocals constraint", inconclusive.reasons)
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

    def test_near_low_vocalness_gets_weak_no_vocals_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            near_low_id = store.upsert_track(track_for(tmp, "near-low.wav"))
            inconclusive_id = store.upsert_track(track_for(tmp, "inconclusive.wav"))
            store.upsert_features(
                TrackFeatures(
                    near_low_id,
                    vocalness=0.38,
                    loudness=-18.0,
                    energy=0.42,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    inconclusive_id,
                    vocalness=0.5,
                    loudness=-18.0,
                    energy=0.42,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            near_low = score_track(store, store.get_track(near_low_id), phase)
            inconclusive = score_track(store, store.get_track(inconclusive_id), phase)

            self.assertGreater(near_low.score, inconclusive.score)
            self.assertIn("vocalness feature weakly supports no-vocals constraint", near_low.reasons)
            store.close()

    def test_vocalness_does_not_dominate_when_vocals_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            low_vocal_id = store.upsert_track(track_for(tmp, "low-vocal.wav"))
            high_vocal_id = store.upsert_track(track_for(tmp, "high-vocal.wav"))
            for track_id, vocalness in ((low_vocal_id, 0.15), (high_vocal_id, 0.9)):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=105.0,
                        loudness=-16.0,
                        energy=0.5,
                        vocalness=vocalness,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )

            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "allow")
            low_vocal = score_track(store, store.get_track(low_vocal_id), phase)
            high_vocal = score_track(store, store.get_track(high_vocal_id), phase)

            self.assertAlmostEqual(low_vocal.score, high_vocal.score)
            self.assertNotIn("vocalness feature supports no-vocals constraint", low_vocal.reasons)
            self.assertNotIn("vocalness feature conflicts with no-vocals constraint", high_vocal.reasons)
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
