import math
import tempfile
import unittest
import wave
from pathlib import Path

from tonepath.analysis import analyze_library, analyze_track_basic
from tonepath.db import TonepathStore
from tonepath.models import Track


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
            features = analyze_track_basic(track_for(path, "song.mp3", track_id=1))
            self.assertEqual(features.confidence, "low")
            self.assertIsNone(features.energy)
            self.assertIsNone(features.loudness)


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


if __name__ == "__main__":
    unittest.main()
