import tempfile
import unittest
from pathlib import Path

from tonepath.analysis import AUDIO_SEPARATOR_FEATURE_SOURCE, ESSENTIA_VOICE_FEATURE_SOURCE
from tonepath.db import TonepathStore
from tonepath.models import EnrichmentRecord, SessionPhase, SessionPlan, SessionRequest, Track, TrackFeatures
from tonepath.selector import score_track, select_path


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

    def test_low_stimulation_soften_demotes_loud_high_energy_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav"))
            loud_id = store.upsert_track(track_for(tmp, "loud.wav"))
            store.upsert_features(
                TrackFeatures(
                    gentle_id,
                    bpm=100.0,
                    loudness=-13.0,
                    energy=0.56,
                    vocalness=0.42,
                    arousal_estimate=0.38,
                    valence_estimate=0.52,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    loud_id,
                    bpm=100.0,
                    loudness=-8.8,
                    energy=0.69,
                    vocalness=0.62,
                    arousal_estimate=0.48,
                    valence_estimate=0.52,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("soften", 0, 600, 0.3, 0.45, 0.3)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            loud = score_track(store, store.get_track(loud_id), phase)

            self.assertGreater(gentle.score, loud.score)
            self.assertIn("low-stimulation safety penalty adjusted the score", loud.reasons)
            store.close()

    def test_low_stimulation_soften_demotes_vocal_heavy_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav"))
            vocal_id = store.upsert_track(track_for(tmp, "vocal.wav"))
            store.upsert_features(
                TrackFeatures(
                    gentle_id,
                    bpm=112.0,
                    loudness=-13.0,
                    energy=0.56,
                    vocalness=0.28,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    vocal_id,
                    bpm=112.0,
                    loudness=-13.0,
                    energy=0.56,
                    vocalness=0.8,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("soften", 0, 600, 0.22, 0.45, 0.22)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            vocal = score_track(store, store.get_track(vocal_id), phase)

            self.assertGreater(gentle.score, vocal.score)
            self.assertIn("vocal-heavy track is risky for low-stimulation phase", vocal.reasons)
            store.close()

    def test_sleep_calm_safety_demotes_moderate_stimulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav"))
            active_id = store.upsert_track(track_for(tmp, "active.wav"))
            store.upsert_features(
                TrackFeatures(
                    gentle_id,
                    bpm=90.0,
                    loudness=-18.0,
                    energy=0.32,
                    vocalness=0.2,
                    arousal_estimate=0.25,
                    valence_estimate=0.45,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    active_id,
                    bpm=129.0,
                    loudness=-17.5,
                    energy=0.43,
                    vocalness=0.56,
                    arousal_estimate=0.32,
                    valence_estimate=0.45,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("soften", 0, 600, 0.3, 0.45, 0.3)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            active = score_track(store, store.get_track(active_id), phase)

            self.assertGreater(gentle.score, active.score)
            self.assertIn("sleep/calm safety penalty adjusted the score", active.reasons)
            store.close()

    def test_sleep_calm_semantic_risk_explains_vocal_allegro_dramatic_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav", title="Soft Piano"))
            risky_id = store.upsert_track(
                track_for(tmp, "risky.wav", title="Choral Allegro Dramatic Piece")
            )
            for track_id in (gentle_id, risky_id):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=96.0,
                        loudness=-18.0,
                        energy=0.35,
                        vocalness=0.25,
                        arousal_estimate=0.28,
                        valence_estimate=0.42,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )

            phase = SessionPhase("calm", 0, 600, 0.2, 0.55, 0.2)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            risky = score_track(store, store.get_track(risky_id), phase)

            self.assertGreater(gentle.score, risky.score)
            self.assertIn("semantic risk: choral_or_vocal_ensemble for low-stimulation phase", risky.reasons)
            self.assertIn("semantic risk: vocal ensemble for sleep/calm", risky.reasons)
            self.assertIn("semantic risk: allegro/showpiece for sleep/calm", risky.reasons)
            store.close()

    def test_energized_phase_does_not_apply_sleep_calm_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            active_id = store.upsert_track(track_for(tmp, "active.wav", title="Active Allegro"))
            store.upsert_features(
                TrackFeatures(
                    active_id,
                    bpm=132.0,
                    loudness=-10.0,
                    energy=0.72,
                    vocalness=0.58,
                    arousal_estimate=0.62,
                    valence_estimate=0.7,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("energize", 0, 600, 0.7, 0.65, 0.75)
            active = score_track(store, store.get_track(active_id), phase)

            self.assertNotIn("sleep/calm safety penalty adjusted the score", active.reasons)
            self.assertNotIn("semantic risk: allegro/showpiece for sleep/calm", active.reasons)
            store.close()

    def test_low_stimulation_hold_demotes_fast_low_vocalness_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav"))
            fast_id = store.upsert_track(track_for(tmp, "fast.wav"))
            store.upsert_features(
                TrackFeatures(
                    gentle_id,
                    bpm=96.0,
                    loudness=-14.0,
                    energy=0.56,
                    vocalness=0.35,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    fast_id,
                    bpm=144.0,
                    loudness=-22.0,
                    energy=0.28,
                    vocalness=0.12,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("hold", 0, 600, 0.24, 0.48, 0.24)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            fast = score_track(store, store.get_track(fast_id), phase)

            self.assertGreater(gentle.score, fast.score)
            self.assertIn("low-stimulation safety penalty adjusted the score", fast.reasons)
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
            self.assertIn("inconclusive vocalness is risky for strict no-vocals constraint", inconclusive.reasons)
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

    def test_select_path_deduplicates_canonical_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            first_id = store.upsert_track(track_for(tmp, "duplicate-one.wav", title="A Serene Garden", artist="Composer"))
            second_id = store.upsert_track(track_for(tmp, "duplicate-two.wav", title="A Serene Garden(null)", artist="Composer"))
            other_id = store.upsert_track(track_for(tmp, "other.wav", title="Different", artist="Composer"))
            for track_id in (first_id, second_id, other_id):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=92.0,
                        loudness=-16.0,
                        energy=0.4,
                        vocalness=0.2,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )
            phase = SessionPhase("focus", 0, 600, 0.5, 0.6, 0.5, "avoid")
            plan = SessionPlan(SessionRequest("focus", "unspecified", "focus", 1800, no_vocals=True), (phase,))

            candidates = select_path(store, plan, limit_per_phase=3)

            titles = [candidate.track.title for candidate in candidates]
            self.assertEqual(len(candidates), 2)
            self.assertIn("Different", titles)
            self.assertEqual(sum(1 for title in titles if title and "A Serene Garden" in title), 1)
            store.close()

    def test_lift_phase_demotes_sad_dark_affect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            warm_id = store.upsert_track(track_for(tmp, "warm-uplift.wav"))
            dark_id = store.upsert_track(track_for(tmp, "dark-sad.wav"))
            for track_id, valence in ((warm_id, 0.72), (dark_id, 0.42)):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=92.0,
                        loudness=-18.0,
                        energy=0.35,
                        vocalness=0.2,
                        arousal_estimate=0.35,
                        valence_estimate=valence,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )
            upsert_affect(store, warm_id, uplift=0.7, warmth=0.7, brightness=0.5, sadness=0.1, darkness=0.1, tension=0.1)
            upsert_affect(store, dark_id, uplift=0.1, warmth=0.2, brightness=0.1, sadness=0.8, darkness=0.7, tension=0.4)

            phase = SessionPhase("lift", 0, 600, 0.42, 0.68, 0.45, "avoid")
            warm = score_track(store, store.get_track(warm_id), phase)
            dark = score_track(store, store.get_track(dark_id), phase)

            self.assertGreater(warm.score, dark.score)
            self.assertIn("affect profile contributes to phase fit", warm.reasons)
            self.assertIn("sad/dark/tension affect is risky for the lift phase", dark.reasons)
            store.close()

    def test_gentle_lift_prefers_safe_higher_valence_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            low_id = store.upsert_track(track_for(tmp, "quiet-low.wav", title="Quiet Low"))
            uplift_id = store.upsert_track(track_for(tmp, "gentle-uplift.wav", title="Gentle Uplift"))
            store.upsert_features(
                TrackFeatures(
                    low_id,
                    bpm=104.0,
                    loudness=-18.0,
                    energy=0.36,
                    vocalness=0.18,
                    arousal_estimate=0.35,
                    valence_estimate=0.42,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    uplift_id,
                    bpm=104.0,
                    loudness=-11.5,
                    energy=0.62,
                    vocalness=0.28,
                    arousal_estimate=0.42,
                    valence_estimate=0.56,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            upsert_affect(store, uplift_id, uplift=0.6, warmth=0.55, calmness=0.35, sadness=0.2, darkness=0.1, tension=0.15)

            phase = SessionPhase("lift", 0, 600, 0.42, 0.68, 0.45)
            low = score_track(store, store.get_track(low_id), phase)
            uplift = score_track(store, store.get_track(uplift_id), phase)

            self.assertGreater(uplift.score, low.score)
            self.assertIn("uplift phase valence fit adjusted the score", uplift.reasons)
            self.assertIn("uplift phase valence is low for gentle lift", low.reasons)
            store.close()

    def test_gentle_lift_does_not_reward_unsafe_high_valence_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            safe_id = store.upsert_track(track_for(tmp, "safe-uplift.wav", title="Safe Uplift"))
            unsafe_id = store.upsert_track(track_for(tmp, "loud-vocal-uplift.wav", title="Loud Vocal Uplift"))
            store.upsert_features(
                TrackFeatures(
                    safe_id,
                    bpm=104.0,
                    loudness=-16.0,
                    energy=0.48,
                    vocalness=0.24,
                    arousal_estimate=0.4,
                    valence_estimate=0.54,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            store.upsert_features(
                TrackFeatures(
                    unsafe_id,
                    bpm=132.0,
                    loudness=-6.5,
                    energy=0.78,
                    vocalness=0.86,
                    arousal_estimate=0.62,
                    valence_estimate=0.74,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("lift", 0, 600, 0.42, 0.68, 0.45, "avoid")
            safe = score_track(store, store.get_track(safe_id), phase)
            unsafe = score_track(store, store.get_track(unsafe_id), phase)

            self.assertGreater(safe.score, unsafe.score)
            self.assertIn("uplift phase valence fit adjusted the score", safe.reasons)
            self.assertNotIn("uplift phase valence fit adjusted the score", unsafe.reasons)
            store.close()

    def test_energized_phase_does_not_apply_gentle_lift_valence_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "upbeat.db")
            upbeat_id = store.upsert_track(track_for(tmp, "upbeat.wav", title="Upbeat"))
            store.upsert_features(
                TrackFeatures(
                    upbeat_id,
                    bpm=124.0,
                    loudness=-10.0,
                    energy=0.72,
                    vocalness=0.3,
                    arousal_estimate=0.62,
                    valence_estimate=0.74,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("energize", 0, 600, 0.7, 0.65, 0.75)
            upbeat = score_track(store, store.get_track(upbeat_id), phase)

            self.assertNotIn("uplift phase valence fit adjusted the score", upbeat.reasons)
            self.assertNotIn("uplift phase valence is low for gentle lift", upbeat.reasons)
            store.close()

    def test_hold_phase_demotes_high_bpm_low_stimulation_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            steady_id = store.upsert_track(track_for(tmp, "steady-warm.wav"))
            frantic_id = store.upsert_track(track_for(tmp, "frantic-warm.wav"))
            for track_id, bpm in ((steady_id, 92.0), (frantic_id, 148.0)):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=bpm,
                        loudness=-15.0,
                        energy=0.42,
                        vocalness=0.15,
                        arousal_estimate=0.38,
                        valence_estimate=0.56,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )
                upsert_affect(store, track_id, uplift=0.45, warmth=0.5, calmness=0.45, sadness=0.2, darkness=0.1, tension=0.1)

            phase = SessionPhase("hold", 0, 600, 0.22, 0.48, 0.24, "allow")
            steady = score_track(store, store.get_track(steady_id), phase)
            frantic = score_track(store, store.get_track(frantic_id), phase)

            self.assertGreater(steady.score, frantic.score)
            self.assertIn("phase stimulation penalty adjusted the score", frantic.reasons)
            store.close()

    def test_calm_phase_demotes_march_like_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav", title="Gentle Piano"))
            march_id = store.upsert_track(track_for(tmp, "march.wav", title="Marche Militaire No. 1"))
            for track_id in (gentle_id, march_id):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=104.0,
                        loudness=-19.0,
                        energy=0.35,
                        vocalness=0.16,
                        arousal_estimate=0.3,
                        valence_estimate=0.36,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )

            phase = SessionPhase("calm", 0, 600, 0.2, 0.6, 0.2)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            march = score_track(store, store.get_track(march_id), phase)

            self.assertGreater(gentle.score, march.score)
            self.assertIn("semantic risk: march_like for low-stimulation phase", march.reasons)
            store.close()

    def test_calm_phase_demotes_voice_ensemble_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            gentle_id = store.upsert_track(track_for(tmp, "gentle.wav", title="Soft Piano"))
            voice_id = store.upsert_track(track_for(tmp, "voice.wav", title="Soft Ensemble"))
            for track_id in (gentle_id, voice_id):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=104.0,
                        loudness=-18.0,
                        energy=0.42,
                        vocalness=0.55,
                        arousal_estimate=0.35,
                        valence_estimate=0.42,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )
            upsert_tag(store, voice_id, "voice", 0.72)

            phase = SessionPhase("settle", 0, 600, 0.25, 0.55, 0.25)
            gentle = score_track(store, store.get_track(gentle_id), phase)
            voice = score_track(store, store.get_track(voice_id), phase)

            self.assertGreater(gentle.score, voice.score)
            self.assertIn("semantic risk: choral_or_vocal_ensemble for low-stimulation phase", voice.reasons)
            store.close()

    def test_malformed_none_tag_score_does_not_trigger_semantic_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = store.upsert_track(track_for(tmp, "soft.wav", title="Soft Ensemble"))
            store.upsert_features(
                TrackFeatures(
                    track_id,
                    bpm=104.0,
                    loudness=-18.0,
                    energy=0.42,
                    vocalness=0.5,
                    arousal_estimate=0.35,
                    valence_estimate=0.42,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )
            original_list_enrichment = store.list_enrichment
            store.list_enrichment = lambda _: [
                EnrichmentRecord(
                    track_id=track_id,
                    field="tag:voice",
                    value=None,  # type: ignore[arg-type]
                    tier="features",
                    source="test",
                    confidence="low",
                )
            ]

            phase = SessionPhase("settle", 0, 600, 0.25, 0.55, 0.25)
            try:
                candidate = score_track(store, store.get_track(track_id), phase)
            finally:
                store.list_enrichment = original_list_enrichment

            self.assertNotIn("semantic risk: choral_or_vocal_ensemble for low-stimulation phase", candidate.reasons)
            store.close()

    def test_low_energy_stabilize_demotes_voice_ensemble_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            steady_id = store.upsert_track(track_for(tmp, "steady.wav", title="Steady Piano"))
            voice_id = store.upsert_track(track_for(tmp, "voice.wav", title="Steady Voices"))
            for track_id in (steady_id, voice_id):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=104.0,
                        loudness=-18.0,
                        energy=0.42,
                        vocalness=0.5,
                        arousal_estimate=0.35,
                        valence_estimate=0.42,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )
            upsert_tag(store, voice_id, "voice", 0.72)

            phase = SessionPhase("stabilize", 0, 600, 0.35, 0.58, 0.38)
            steady = score_track(store, store.get_track(steady_id), phase)
            voice = score_track(store, store.get_track(voice_id), phase)

            self.assertGreater(steady.score, voice.score)
            self.assertIn("semantic risk: choral_or_vocal_ensemble for low-stimulation phase", voice.reasons)
            store.close()

    def test_low_energy_lift_demotes_epic_dramatic_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            warm_id = store.upsert_track(track_for(tmp, "warm.wav", title="Warm Lift"))
            epic_id = store.upsert_track(track_for(tmp, "epic.wav", title="Dramatic Lift"))
            for track_id in (warm_id, epic_id):
                store.upsert_features(
                    TrackFeatures(
                        track_id,
                        bpm=104.0,
                        loudness=-17.0,
                        energy=0.42,
                        vocalness=0.2,
                        arousal_estimate=0.38,
                        valence_estimate=0.55,
                        feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                        confidence="high",
                    )
                )
                upsert_affect(store, track_id, uplift=0.55, warmth=0.5, calmness=0.4, sadness=0.2, darkness=0.1, tension=0.15)
            upsert_tag(store, epic_id, "epic", 0.7)

            phase = SessionPhase("lift", 0, 600, 0.42, 0.68, 0.45)
            warm = score_track(store, store.get_track(warm_id), phase)
            epic = score_track(store, store.get_track(epic_id), phase)

            self.assertGreater(warm.score, epic.score)
            self.assertIn("semantic risk: epic_or_dramatic for low-stimulation phase", epic.reasons)
            store.close()

    def test_energized_phase_does_not_apply_low_stimulation_semantic_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            march_id = store.upsert_track(track_for(tmp, "march.wav", title="Radetzky March"))
            store.upsert_features(
                TrackFeatures(
                    march_id,
                    bpm=120.0,
                    loudness=-12.0,
                    energy=0.72,
                    vocalness=0.3,
                    arousal_estimate=0.6,
                    valence_estimate=0.62,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("energize", 0, 600, 0.7, 0.65, 0.75)
            march = score_track(store, store.get_track(march_id), phase)

            self.assertNotIn("semantic risk: march_like for low-stimulation phase", march.reasons)
            store.close()

    def test_unverified_audio_without_duration_or_features_is_demoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            verified_id = store.upsert_track(track_for(tmp, "verified.wav"))
            unverified_id = store.upsert_track(track_for(tmp, "broken.mp3"))
            store.upsert_features(
                TrackFeatures(
                    verified_id,
                    bpm=92.0,
                    loudness=-16.0,
                    energy=0.4,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            phase = SessionPhase("calm", 0, 600, 0.2, 0.5, 0.2)
            verified = score_track(store, store.get_track(verified_id), phase)
            unverified = score_track(store, store.get_track(unverified_id), phase)

            self.assertGreater(verified.score, unverified.score)
            self.assertIn("low-evidence/unverified audio candidate", unverified.reasons)
            store.close()


def track_for(tmp: str, name: str, title: str | None = None, artist: str | None = "artist") -> Track:
    path = Path(tmp) / name
    path.write_bytes(b"not real audio")
    return Track(
        id=None,
        path=path,
        file_hash=name,
        mtime=1.0,
        title=title or name,
        artist=artist,
        album=None,
        genre=None,
        duration=None,
        format="wav",
    )


def upsert_affect(store: TonepathStore, track_id: int, **values: float) -> None:
    """Store derived affect profile values for selector tests."""

    for axis, value in values.items():
        store.upsert_enrichment(
            EnrichmentRecord(
                track_id=track_id,
                field=f"affect:{axis}",
                value=str(value),
                tier="features",
                source="test",
                confidence="medium",
            )
        )


def upsert_tag(store: TonepathStore, track_id: int, label: str, value: float) -> None:
    """Store one model tag value for selector tests."""

    store.upsert_enrichment(
        EnrichmentRecord(
            track_id=track_id,
            field=f"tag:{label}",
            value=str(value),
            tier="features",
            source="test",
            confidence="high",
        )
    )


if __name__ == "__main__":
    unittest.main()
