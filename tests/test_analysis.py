import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from tonepath.analysis import (
    analyze_library,
    analyze_track_basic,
    analyze_track_vocalness,
    analyze_with_ffmpeg,
    estimate_bpm,
    estimate_vocalness,
    loudness_to_unit,
)
from tonepath.db import TonepathStore
from tonepath.models import Track, TrackFeatures


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
