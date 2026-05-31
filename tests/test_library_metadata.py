import json
import os
import tempfile
import unittest
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.display import METADATA_ARTIST_FIELD, METADATA_TITLE_FIELD, display_label
from tonepath.library import library_issues
from tonepath.models import Track, TrackFeatures
from tonepath.readiness import library_status


class LibraryMetadataTest(unittest.TestCase):
    def test_library_issues_reports_dirty_metadata_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch_home(tmp):
                store = TonepathStore()
                first_id = store.upsert_track(track_for(Path(tmp) / "one.mp3", title="Song(null)", artist=None))
                second_id = store.upsert_track(track_for(Path(tmp) / "two.mp3", title="Clean", artist="Artist"))
                duplicate_id = store.upsert_track(track_for(Path(tmp) / "dup.mp3", title="Clean!", artist="Artist"))
                store.upsert_metadata_override(first_id, METADATA_TITLE_FIELD, "Song")
                payload = library_issues(store)
                store.close()

        self.assertEqual(payload["summary"]["dirty_metadata"], 1)
        self.assertEqual(payload["summary"]["duplicate_groups"], 1)
        dirty = payload["dirty_metadata"][0]
        self.assertEqual(dirty["track_id"], first_id)
        self.assertTrue(dirty["has_override"])
        duplicate_ids = {row["track_id"] for row in payload["duplicate_groups"][0]}
        self.assertEqual(duplicate_ids, {second_id, duplicate_id})

    def test_set_meta_updates_effective_display_and_status_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch_home(tmp):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "bad.mp3", title="Bad(null)", artist="unknown"))
                store.upsert_features(features_for(track_id))
                self.assertEqual(library_status(store).dirty_metadata, 1)
                store.close()

                result = CliRunner().invoke(
                    app,
                    ["library", "set-meta", str(track_id), "--title", "Better Title", "--artist", "Better Artist"],
                )
                self.assertEqual(result.exit_code, 0, result.output)

                store = TonepathStore()
                raw = store.get_track(track_id)
                effective = store.get_track(track_id, effective_metadata=True)
                rows = store.list_enrichment(track_id)
                self.assertEqual(raw.title, "Bad(null)")
                self.assertIsNotNone(effective)
                self.assertEqual(display_label(effective), "Better Title - Better Artist")
                self.assertEqual(library_status(store).dirty_metadata, 0)
                self.assertEqual({row.field for row in rows}, {METADATA_TITLE_FIELD, METADATA_ARTIST_FIELD})
                store.close()

    def test_clear_meta_restores_raw_display_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch_home(tmp):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "bad.mp3", title="Bad(null)", artist="unknown"))
                store.upsert_metadata_override(track_id, METADATA_TITLE_FIELD, "Better Title")
                store.upsert_metadata_override(track_id, METADATA_ARTIST_FIELD, "Better Artist")
                self.assertEqual(library_status(store).dirty_metadata, 0)
                store.close()

                result = CliRunner().invoke(app, ["library", "clear-meta", str(track_id)])
                self.assertEqual(result.exit_code, 0, result.output)

                store = TonepathStore()
                self.assertEqual(store.metadata_overrides(track_id), {})
                self.assertEqual(library_status(store).dirty_metadata, 1)
                self.assertEqual(store.profile_summary()["tracks"], 1)
                store.close()

    def test_override_can_resolve_duplicate_status_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch_home(tmp):
                store = TonepathStore()
                first_id = store.upsert_track(track_for(Path(tmp) / "one.mp3", title="Same", artist="Artist"))
                second_id = store.upsert_track(track_for(Path(tmp) / "two.mp3", title="Same!", artist="Artist"))
                store.upsert_features(features_for(first_id))
                store.upsert_features(features_for(second_id))
                self.assertEqual(library_status(store).duplicate_tracks, 1)
                store.upsert_metadata_override(second_id, METADATA_TITLE_FIELD, "Different")
                self.assertEqual(library_status(store).duplicate_tracks, 0)
                store.close()

    def test_library_issues_json_cli_outputs_stable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch_home(tmp):
                store = TonepathStore()
                track_id = store.upsert_track(track_for(Path(tmp) / "bad.mp3", title=None, artist="unknown"))
                store.close()

                result = CliRunner().invoke(app, ["library", "issues", "--json"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["dirty_metadata"][0]["track_id"], track_id)
        self.assertIn("relative_path", payload["dirty_metadata"][0])


def patch_home(tmp: str) -> AbstractContextManager[dict[str, str]]:
    """Return a patched environment for one isolated Tonepath home."""

    return patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")})


def track_for(path: Path, title: str | None, artist: str | None, duration: float | None = 180.0) -> Track:
    """Create one library-metadata test track payload."""

    path.write_bytes(b"fake audio")
    return Track(
        id=None,
        path=path,
        file_hash=path.name,
        mtime=1.0,
        title=title,
        artist=artist,
        album=None,
        genre=None,
        duration=duration,
        format="mp3",
    )


def features_for(track_id: int) -> TrackFeatures:
    """Return complete enough features for readiness tests."""

    return TrackFeatures(
        track_id=track_id,
        bpm=90.0,
        loudness=-16.0,
        energy=0.35,
        vocalness=0.2,
        arousal_estimate=0.3,
        valence_estimate=0.5,
        feature_source="test",
        confidence="high",
    )


if __name__ == "__main__":
    unittest.main()
