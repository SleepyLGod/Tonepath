import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from tonepath.analysis import (
    ESSENTIA_MIR_FEATURE_SOURCE,
    ESSENTIA_TF_TAGS_FEATURE_SOURCE,
    ESSENTIA_TAGS_FEATURE_SOURCE,
    ESSENTIA_VOICE_FEATURE_SOURCE,
    analyze_library,
    analyze_track_mir,
    analyze_track_tags,
    analyze_track_basic,
    analyze_track_vocalness,
    analyze_with_ffmpeg,
    analyze_vocalness_with_audio_separator,
    analyze_vocalness_with_demucs,
    estimate_bpm,
    estimate_vocalness,
    loudness_to_unit,
)
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures
from tonepath.scanner import read_track


class AnalysisTest(unittest.TestCase):
    def test_basic_analysis_extracts_wave_loudness_and_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wave(path, amplitude=12000)
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = store.upsert_track(track_for(path, "tone.wav"))

            analyzed, skipped = analyze_library(store)

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertIsNotNone(features)
            self.assertEqual(features.feature_source, "basic-local-analysis")
            self.assertEqual(features.confidence, "medium")
            self.assertIsNotNone(features.loudness)
            self.assertIsNotNone(features.energy)
            self.assertIsNone(features.bpm)
            store.close()

    def test_basic_analysis_skips_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            missing = Path(tmp) / "missing.wav"
            store.upsert_track(track_for(missing, "missing.wav"))

            analyzed, skipped = analyze_library(store)

            self.assertEqual(analyzed, 0)
            self.assertEqual(skipped, 1)
            store.close()

    def test_basic_analysis_updates_existing_feature_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wave(path, amplitude=8000)
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = store.upsert_track(track_for(path, "tone.wav"))

            analyze_library(store)
            analyze_library(store)

            row = store.conn.execute("SELECT COUNT(*) AS count FROM track_features WHERE track_id = ?", (track_id,)).fetchone()
            self.assertEqual(int(row["count"]), 1)
            store.close()

    def test_non_wave_analysis_is_partial_low_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"not decoded as audio")
            with patch("tonepath.analysis.shutil.which", return_value=None):
                features = analyze_track_basic(track_for(path, "song.mp3", track_id=1))
            self.assertEqual(features.confidence, "low")
            self.assertIsNone(features.energy)
            self.assertIsNone(features.loudness)

    def test_ffmpeg_analysis_extracts_mean_volume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            result = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stderr="[Parsed_volumedetect_0] mean_volume: -18.4 dB\n",
            )
            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
                "tonepath.analysis.subprocess.run",
                return_value=result,
            ):
                features = analyze_track_basic(track_for(path, "song.mp3", track_id=1))
            self.assertEqual(features.confidence, "medium")
            self.assertEqual(features.loudness, -18.4)
            self.assertIsNotNone(features.energy)
            self.assertIsNone(features.bpm)

    def test_ffmpeg_analysis_stores_bpm_when_pcm_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            volume = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stderr="[Parsed_volumedetect_0] mean_volume: -18.4 dB\n",
            )
            pcm = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=samples_to_pcm_bytes(pulse_samples(11025, 120.0, 20.0)),
            )
            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
                "tonepath.analysis.subprocess.run",
                side_effect=[volume, pcm],
            ):
                features = analyze_track_basic(track_for(path, "song.mp3", track_id=1))
            self.assertEqual(features.confidence, "medium")
            self.assertIsNotNone(features.bpm)
            self.assertAlmostEqual(features.bpm, 120.0, delta=15.0)

    def test_estimate_bpm_returns_none_for_ambiguous_signal(self) -> None:
        self.assertIsNone(estimate_bpm([0] * 11025 * 12, 11025))

    def test_estimate_vocalness_separates_voiced_and_non_vocal_signals(self) -> None:
        voiced = voiced_like_samples(11025, 12.0)
        low_drone = sine_samples(11025, 100.0, 12.0)
        percussive = high_percussive_samples(11025, 12.0)

        voiced_score = estimate_vocalness(voiced, 11025)
        low_score = estimate_vocalness(low_drone, 11025)
        percussive_score = estimate_vocalness(percussive, 11025)

        self.assertIsNotNone(voiced_score)
        self.assertIsNotNone(low_score)
        self.assertIsNotNone(percussive_score)
        self.assertGreater(voiced_score, 0.65)
        self.assertLess(low_score, 0.35)
        self.assertLess(percussive_score, 0.35)

    def test_vocalness_analysis_preserves_existing_basic_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            track_id = store.upsert_track(track_for(path, "song.mp3"))
            store.upsert_features(
                TrackFeatures(
                    track_id=track_id,
                    bpm=120.0,
                    loudness=-18.0,
                    energy=0.42,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )

            with patch("tonepath.analysis.decode_pcm_with_ffmpeg", return_value=voiced_like_samples(11025, 12.0)):
                analyzed, skipped = analyze_library(store, features="vocalness")

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertEqual(features.bpm, 120.0)
            self.assertEqual(features.loudness, -18.0)
            self.assertEqual(features.energy, 0.42)
            self.assertIsNotNone(features.vocalness)
            self.assertGreater(features.vocalness, 0.65)
            row = store.conn.execute("SELECT COUNT(*) AS count FROM track_features WHERE track_id = ?", (track_id,)).fetchone()
            self.assertEqual(int(row["count"]), 1)
            store.close()

    def test_basic_analysis_preserves_existing_vocalness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            write_wave(path, amplitude=9000)
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = store.upsert_track(track_for(path, "tone.wav"))
            store.upsert_features(
                TrackFeatures(
                    track_id=track_id,
                    vocalness=0.22,
                    feature_source="basic-local-analysis",
                    confidence="medium",
                )
            )

            analyzed, skipped = analyze_library(store, features="basic")

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertIsNotNone(features.energy)
            self.assertIsNotNone(features.loudness)
            self.assertEqual(features.vocalness, 0.22)
            row = store.conn.execute("SELECT COUNT(*) AS count FROM track_features WHERE track_id = ?", (track_id,)).fetchone()
            self.assertEqual(int(row["count"]), 1)
            store.close()

    def test_vocalness_analysis_keeps_unknown_when_decode_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            existing = TrackFeatures(
                track_id=1,
                bpm=100.0,
                loudness=-20.0,
                energy=0.3,
                feature_source="basic-local-analysis",
                confidence="medium",
            )
            with patch("tonepath.analysis.decode_pcm_with_ffmpeg", return_value=None):
                features = analyze_track_vocalness(track_for(path, "song.mp3", track_id=1), existing)
            self.assertIsNone(features.vocalness)
            self.assertEqual(features.bpm, 100.0)
            self.assertEqual(features.loudness, -20.0)
            self.assertEqual(features.energy, 0.3)

    def test_demucs_method_requires_external_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=9000)
            store.upsert_track(track_for(path, "song.wav"))

            with patch("tonepath.analysis.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "demucs"):
                    analyze_library(store, features="vocalness", method="demucs-cli")
            row = store.conn.execute("SELECT COUNT(*) AS count FROM track_features").fetchone()
            self.assertEqual(int(row["count"]), 0)
            store.close()

    def test_audio_separator_method_requires_optional_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=9000)
            store.upsert_track(track_for(path, "song.wav"))

            with patch("tonepath.analysis.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "uv sync --extra models"):
                    analyze_library(store, features="vocalness", method="audio-separator")
            row = store.conn.execute("SELECT COUNT(*) AS count FROM track_features").fetchone()
            self.assertEqual(int(row["count"]), 0)
            store.close()

    def test_mir_analysis_writes_features_and_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            track_id = store.upsert_track(track_for(path, "song.mp3"))

            with patch("tonepath.analysis.import_essentia_standard", return_value=object()), patch(
                "tonepath.analysis.extract_mir_with_essentia",
                return_value={
                    "bpm": 112.4,
                    "loudness": -14.0,
                    "key": "F#",
                    "scale": "major",
                    "key_strength": 0.86,
                    "danceability": 1.2,
                    "dynamic_complexity": 3.5,
                },
            ):
                analyzed, skipped = analyze_library(store, features="mir", method="essentia")

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertEqual(features.feature_source, ESSENTIA_MIR_FEATURE_SOURCE)
            self.assertEqual(features.confidence, "high")
            self.assertEqual(features.bpm, 112.4)
            self.assertEqual(features.loudness, -14.0)
            self.assertIsNotNone(features.energy)
            enrichment = store.list_enrichment(track_id)
            fields = {record.field: record.value for record in enrichment}
            self.assertEqual(fields["key"], "F#")
            self.assertEqual(fields["scale"], "major")
            self.assertEqual(fields["danceability"], "1.2")
            store.close()

    def test_mir_analysis_requires_essentia_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            with patch("tonepath.analysis.import_essentia_standard", side_effect=RuntimeError("uv sync --extra mir")):
                with self.assertRaisesRegex(RuntimeError, "uv sync --extra mir"):
                    analyze_track_mir(track_for(path, "song.mp3", track_id=1))

    def test_mir_analysis_preserves_voice_classifier_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            existing = TrackFeatures(
                track_id=1,
                vocalness=0.2,
                feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                confidence="high",
            )
            with patch("tonepath.analysis.extract_mir_with_essentia", return_value={"bpm": 120.0, "loudness": -12.0}):
                features, _enrichment = analyze_track_mir(track_for(path, "song.mp3", track_id=1), existing)

            self.assertEqual(features.feature_source, ESSENTIA_VOICE_FEATURE_SOURCE)
            self.assertEqual(features.vocalness, 0.2)
            self.assertEqual(features.bpm, 120.0)

    def test_tag_analysis_maps_voice_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            track_id = store.upsert_track(track_for(path, "song.mp3"))
            store.upsert_features(
                TrackFeatures(
                    track_id=track_id,
                    bpm=100.0,
                    loudness=-16.0,
                    energy=0.4,
                    vocalness=0.7,
                    feature_source="model-audio-separator",
                    confidence="high",
                )
            )

            with patch(
                "tonepath.analysis.ensure_essentia_tagging_available",
                return_value=None,
            ), patch(
                "tonepath.analysis.extract_tags_with_essentia",
                return_value={
                    "vocalness": 0.18,
                    "tags": [("mood/theme---focus", 0.82), ("instrument---piano", 0.61)],
                },
            ):
                analyzed, skipped = analyze_library(store, features="tags", method="essentia", force=True)

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertEqual(features.feature_source, ESSENTIA_VOICE_FEATURE_SOURCE)
            self.assertEqual(features.vocalness, 0.18)
            self.assertEqual(features.bpm, 100.0)
            enrichment = store.list_enrichment(track_id)
            fields = {record.field: record.value for record in enrichment}
            self.assertEqual(fields["tag:mood/theme---focus"], "0.82")
            self.assertEqual(fields["tag:instrument---piano"], "0.61")
            self.assertTrue(all(record.source == ESSENTIA_TAGS_FEATURE_SOURCE for record in enrichment))
            store.close()

    def test_essentia_tf_tag_analysis_maps_voice_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"fake")
            track_id = store.upsert_track(track_for(path, "song.mp3"))

            with patch("tonepath.analysis.ensure_essentia_tf_runtime", return_value=None), patch(
                "tonepath.analysis.run_essentia_tf_tags",
                return_value={
                    "vocalness": 0.12,
                    "tags": [["voice", 0.1], ["instrumental", 0.9], ["mood/theme---calm", 0.72]],
                },
            ):
                analyzed, skipped = analyze_library(store, features="tags", method="essentia-tf")

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertEqual(features.feature_source, ESSENTIA_VOICE_FEATURE_SOURCE)
            self.assertEqual(features.vocalness, 0.12)
            enrichment = store.list_enrichment(track_id)
            self.assertTrue(any(record.source == ESSENTIA_TF_TAGS_FEATURE_SOURCE for record in enrichment))
            store.close()

    def test_separator_does_not_overwrite_essentia_voice_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=9000)
            track_id = store.upsert_track(track_for(path, "song.wav"))
            store.upsert_features(
                TrackFeatures(
                    track_id=track_id,
                    vocalness=0.12,
                    feature_source=ESSENTIA_VOICE_FEATURE_SOURCE,
                    confidence="high",
                )
            )

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.analyze_vocalness_with_audio_separator",
                return_value=0.9,
            ):
                analyzed, skipped = analyze_library(store, features="vocalness", method="audio-separator")

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 0)
            self.assertEqual(skipped, 1)
            self.assertEqual(features.vocalness, 0.12)
            self.assertEqual(features.feature_source, ESSENTIA_VOICE_FEATURE_SOURCE)
            store.close()

    def test_audio_separator_method_writes_high_confidence_vocalness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "song.wav"
            write_wave(path, amplitude=12000)
            track = track_for(path, "song.wav", track_id=1)
            existing = TrackFeatures(
                track_id=1,
                bpm=118.0,
                loudness=-16.0,
                energy=0.5,
                feature_source="basic-local-analysis",
                confidence="medium",
            )

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
                output_root = Path(command[command.index("--output_dir") + 1])
                output_root.mkdir(parents=True, exist_ok=True)
                write_wave(output_root / "song_(Vocals).wav", amplitude=6000)
                return subprocess.CompletedProcess(command, returncode=0)

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.config.ensure_data_dir",
                return_value=root,
            ), patch("tonepath.analysis.subprocess.run", side_effect=fake_run):
                features = analyze_track_vocalness(track, existing, method="audio-separator")

            self.assertEqual(features.feature_source, "model-audio-separator")
            self.assertEqual(features.confidence, "high")
            self.assertEqual(features.bpm, 118.0)
            self.assertIsNotNone(features.vocalness)
            self.assertGreater(features.vocalness, 0.3)
            self.assertLess(features.vocalness, 0.8)

    def test_audio_separator_failure_preserves_existing_vocalness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "song.wav"
            write_wave(path, amplitude=12000)
            track = track_for(path, "song.wav", track_id=1)
            existing = TrackFeatures(
                track_id=1,
                bpm=118.0,
                loudness=-16.0,
                energy=0.5,
                vocalness=0.22,
                feature_source="basic-local-analysis",
                confidence="medium",
            )
            result = subprocess.CompletedProcess(args=[], returncode=1)

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.config.ensure_data_dir",
                return_value=root,
            ), patch("tonepath.analysis.subprocess.run", return_value=result):
                features = analyze_track_vocalness(track, existing, method="audio-separator")

            self.assertEqual(features.vocalness, 0.22)
            self.assertEqual(features.feature_source, "basic-local-analysis")
            self.assertEqual(features.confidence, "medium")

    def test_audio_separator_helper_returns_none_when_stem_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=12000)
            result = subprocess.CompletedProcess(args=[], returncode=0)
            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.config.ensure_data_dir",
                return_value=Path(tmp),
            ), patch("tonepath.analysis.subprocess.run", return_value=result):
                self.assertIsNone(analyze_vocalness_with_audio_separator(path))

    def test_model_analysis_skips_existing_same_source_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=12000)
            track_id = store.upsert_track(track_for(path, "song.wav"))
            store.upsert_features(
                TrackFeatures(
                    track_id=track_id,
                    vocalness=0.2,
                    feature_source="model-audio-separator",
                    confidence="high",
                )
            )

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.analyze_vocalness_with_audio_separator",
                side_effect=AssertionError("should not re-run"),
            ):
                analyzed, skipped = analyze_library(store, features="vocalness", method="audio-separator")

            self.assertEqual(analyzed, 0)
            self.assertEqual(skipped, 1)
            store.close()

    def test_force_reanalyzes_existing_model_feature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=12000)
            track_id = store.upsert_track(track_for(path, "song.wav"))
            store.upsert_features(
                TrackFeatures(
                    track_id=track_id,
                    vocalness=0.2,
                    feature_source="model-audio-separator",
                    confidence="high",
                )
            )

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.analyze_vocalness_with_audio_separator",
                return_value=0.6,
            ):
                analyzed, skipped = analyze_library(store, features="vocalness", method="audio-separator", force=True)

            features = store.get_features(track_id)
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 0)
            self.assertEqual(features.vocalness, 0.6)
            store.close()

    def test_limit_restricts_eligible_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            for index in range(3):
                path = Path(tmp) / f"song-{index}.wav"
                write_wave(path, amplitude=12000)
                store.upsert_track(track_for(path, f"song-{index}.wav"))

            with patch("tonepath.analysis.decode_wave_pcm", return_value=(voiced_like_samples(11025, 12.0), 11025)):
                analyzed, skipped = analyze_library(store, features="vocalness", limit=2)

            rows = store.conn.execute("SELECT COUNT(*) AS count FROM track_features").fetchone()
            self.assertEqual(analyzed, 2)
            self.assertEqual(skipped, 0)
            self.assertEqual(int(rows["count"]), 2)
            store.close()

    def test_changed_only_processes_changed_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            stable = Path(tmp) / "stable.wav"
            changed = Path(tmp) / "changed.wav"
            write_wave(stable, amplitude=9000)
            write_wave(changed, amplitude=9000)
            store.upsert_track(read_track(stable))
            store.upsert_track(track_for(changed, "changed.wav"))

            analyzed, skipped = analyze_library(store, features="basic", changed_only=True)

            row = store.conn.execute("SELECT COUNT(*) AS count FROM track_features").fetchone()
            self.assertEqual(analyzed, 1)
            self.assertEqual(skipped, 1)
            self.assertEqual(int(row["count"]), 1)
            store.close()

    def test_model_failure_skips_track_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_ids = []
            for index in range(3):
                path = Path(tmp) / f"song-{index}.wav"
                write_wave(path, amplitude=12000)
                track_ids.append(store.upsert_track(track_for(path, f"song-{index}.wav")))

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.analyze_vocalness_with_audio_separator",
                side_effect=[0.2, RuntimeError("separator failed"), 0.8],
            ):
                analyzed, skipped = analyze_library(store, features="vocalness", method="audio-separator")

            self.assertEqual(analyzed, 2)
            self.assertEqual(skipped, 1)
            self.assertEqual(store.get_features(track_ids[0]).vocalness, 0.2)
            self.assertIsNone(store.get_features(track_ids[1]))
            self.assertEqual(store.get_features(track_ids[2]).vocalness, 0.8)
            store.close()

    def test_keyboard_interrupt_keeps_completed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_ids = []
            for index in range(2):
                path = Path(tmp) / f"song-{index}.wav"
                write_wave(path, amplitude=12000)
                track_ids.append(store.upsert_track(track_for(path, f"song-{index}.wav")))

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/audio-separator"), patch(
                "tonepath.analysis.analyze_vocalness_with_audio_separator",
                side_effect=[0.2, KeyboardInterrupt()],
            ):
                with self.assertRaises(KeyboardInterrupt):
                    analyze_library(store, features="vocalness", method="audio-separator")

            self.assertEqual(store.get_features(track_ids[0]).vocalness, 0.2)
            self.assertIsNone(store.get_features(track_ids[1]))
            store.close()

    def test_demucs_method_writes_high_confidence_vocalness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "song.wav"
            write_wave(path, amplitude=12000)
            track = track_for(path, "song.wav", track_id=1)
            existing = TrackFeatures(
                track_id=1,
                bpm=118.0,
                loudness=-16.0,
                energy=0.5,
                feature_source="basic-local-analysis",
                confidence="medium",
            )

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
                cache_root = Path(command[command.index("-o") + 1])
                stem_dir = cache_root / "htdemucs" / "song"
                stem_dir.mkdir(parents=True, exist_ok=True)
                write_wave(stem_dir / "vocals.wav", amplitude=6000)
                return subprocess.CompletedProcess(command, returncode=0)

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/demucs"), patch(
                "tonepath.analysis.config.ensure_data_dir",
                return_value=root,
            ), patch("tonepath.analysis.subprocess.run", side_effect=fake_run):
                features = analyze_track_vocalness(track, existing, method="demucs-cli")

            self.assertEqual(features.feature_source, "model-demucs-cli")
            self.assertEqual(features.confidence, "high")
            self.assertEqual(features.bpm, 118.0)
            self.assertIsNotNone(features.vocalness)
            self.assertGreater(features.vocalness, 0.3)
            self.assertLess(features.vocalness, 0.8)

    def test_demucs_failure_preserves_existing_vocalness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "song.wav"
            write_wave(path, amplitude=12000)
            track = track_for(path, "song.wav", track_id=1)
            existing = TrackFeatures(
                track_id=1,
                bpm=118.0,
                loudness=-16.0,
                energy=0.5,
                vocalness=0.22,
                feature_source="basic-local-analysis",
                confidence="medium",
            )
            result = subprocess.CompletedProcess(args=[], returncode=1)

            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/demucs"), patch(
                "tonepath.analysis.config.ensure_data_dir",
                return_value=root,
            ), patch("tonepath.analysis.subprocess.run", return_value=result):
                features = analyze_track_vocalness(track, existing, method="demucs-cli")

            self.assertEqual(features.vocalness, 0.22)
            self.assertEqual(features.feature_source, "basic-local-analysis")
            self.assertEqual(features.confidence, "medium")

    def test_demucs_vocalness_helper_returns_none_when_stem_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.wav"
            write_wave(path, amplitude=12000)
            result = subprocess.CompletedProcess(args=[], returncode=0)
            with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/demucs"), patch(
                "tonepath.analysis.config.ensure_data_dir",
                return_value=Path(tmp),
            ), patch("tonepath.analysis.subprocess.run", return_value=result):
                self.assertIsNone(analyze_vocalness_with_demucs(path))

    def test_loudness_mapping_has_useful_music_range(self) -> None:
        self.assertLess(loudness_to_unit(-18.0), loudness_to_unit(-9.0))
        self.assertAlmostEqual(loudness_to_unit(-30.0), 0.0)
        self.assertAlmostEqual(loudness_to_unit(0.0), 1.0)

    def test_ffmpeg_analysis_handles_common_audio_suffixes_without_crashing(self) -> None:
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stderr="[Parsed_volumedetect_0] mean_volume: -24.0 dB\n",
        )
        with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
            "tonepath.analysis.subprocess.run",
            return_value=result,
        ):
            for suffix in (".mp3", ".flac", ".m4a", ".wav"):
                with self.subTest(suffix=suffix):
                    features = analyze_with_ffmpeg(Path(f"/tmp/song{suffix}"))
                    self.assertIsNotNone(features)

    def test_ffmpeg_analysis_returns_none_for_unreadable_output(self) -> None:
        result = subprocess.CompletedProcess(args=[], returncode=1, stderr="invalid data")
        with patch("tonepath.analysis.shutil.which", return_value="/usr/bin/ffmpeg"), patch(
            "tonepath.analysis.subprocess.run",
            return_value=result,
        ):
            features = analyze_with_ffmpeg(Path("/tmp/song.mp3"))
        self.assertIsNone(features)


def track_for(path: Path, title: str, track_id: int | None = None) -> Track:
    return Track(
        id=track_id,
        path=path,
        file_hash=title,
        mtime=1.0,
        title=title,
        artist="artist",
        album=None,
        genre=None,
        duration=None,
        format=path.suffix.lstrip("."),
    )


def write_wave(path: Path, amplitude: int) -> None:
    samples = bytearray()
    for index in range(8000):
        value = int(amplitude * math.sin(index / 12.0))
        samples.extend(value.to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(bytes(samples))


def pulse_samples(sample_rate: int, bpm: float, seconds: float) -> list[int]:
    samples = [0] * int(sample_rate * seconds)
    beat_interval = int(sample_rate * 60.0 / bpm)
    pulse_width = max(sample_rate // 25, 1)
    for beat_start in range(0, len(samples), beat_interval):
        for offset in range(pulse_width):
            index = beat_start + offset
            if index < len(samples):
                samples[index] = 18000
    return samples


def voiced_like_samples(sample_rate: int, seconds: float) -> list[int]:
    samples = []
    for index in range(int(sample_rate * seconds)):
        time = index / sample_rate
        amplitude = 0.55 + 0.45 * math.sin(2 * math.pi * 4 * time)
        value = amplitude * (
            math.sin(2 * math.pi * 220 * time)
            + 0.7 * math.sin(2 * math.pi * 440 * time)
            + 0.5 * math.sin(2 * math.pi * 880 * time)
            + 0.3 * math.sin(2 * math.pi * 1760 * time)
        )
        samples.append(int(value * 7000))
    return samples


def sine_samples(sample_rate: int, frequency: float, seconds: float) -> list[int]:
    return [int(math.sin(2 * math.pi * frequency * index / sample_rate) * 10000) for index in range(int(sample_rate * seconds))]


def high_percussive_samples(sample_rate: int, seconds: float) -> list[int]:
    samples = []
    for index in range(int(sample_rate * seconds)):
        time = index / sample_rate
        value = math.sin(2 * math.pi * 4500 * time) * 9000 if int(time * 4) % 4 == 0 else 0.0
        samples.append(int(value))
    return samples


def samples_to_pcm_bytes(samples: list[int]) -> bytes:
    data = bytearray()
    for sample in samples:
        data.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(data)


if __name__ == "__main__":
    unittest.main()
