import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.db import TonepathStore
from tonepath.enrichment import enrich_library
from tonepath.models import Track


class EnrichmentTest(unittest.TestCase):
    def test_local_enrichment_stores_source_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TonepathStore(Path(tmp) / "tonepath.db")
            track_id = store.upsert_track(
                Track(
                    id=None,
                    path=Path(tmp) / "song.mp3",
                    file_hash="hash",
                    mtime=1.0,
                    title="Song",
                    artist="Artist",
                    album=None,
                    genre="ambient",
                    duration=None,
                    format="mp3",
                )
            )
            count = enrich_library(store, "local")
            records = store.list_enrichment(track_id)
            self.assertGreaterEqual(count, 4)
            self.assertTrue(any(record.field == "title" for record in records))
            self.assertTrue(all(record.source == "local-metadata" for record in records))
            self.assertTrue(all(not record.is_online for record in records))
            self.assertTrue(all(record.confidence in {"low", "medium"} for record in records))
            store.close()

    def test_online_enrichment_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                with self.assertRaises(PermissionError):
                    enrich_library(store, "musicbrainz")
                store.close()


if __name__ == "__main__":
    unittest.main()
