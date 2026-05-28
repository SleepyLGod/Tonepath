import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.embedding import clap_audio_embedding_missing, read_clap_audio_embedding, write_clap_audio_embedding
from tonepath.models import Track


class EmbeddingCacheTest(unittest.TestCase):
    def test_audio_embedding_cache_invalidates_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = Path(tmp) / "song.mp3"
            path.write_bytes(b"first")
            track = track_for(path)
            with patch.dict(os.environ, {"TONEPATH_HOME": str(home)}):
                write_clap_audio_embedding(track, {"model_id": "laion-clap-default", "embedding": [0.1, 0.2, 0.3]})
                self.assertEqual(read_clap_audio_embedding(track), [0.1, 0.2, 0.3])
                self.assertFalse(clap_audio_embedding_missing(track))

                path.write_bytes(b"second")

                self.assertIsNone(read_clap_audio_embedding(track))
                self.assertTrue(clap_audio_embedding_missing(track))


def track_for(path: Path) -> Track:
    """Return one persisted track for cache tests."""

    return Track(
        id=1,
        path=path,
        file_hash="hash",
        mtime=path.stat().st_mtime,
        title="song",
        artist="artist",
        album=None,
        genre=None,
        duration=180.0,
        format="mp3",
    )


if __name__ == "__main__":
    unittest.main()
