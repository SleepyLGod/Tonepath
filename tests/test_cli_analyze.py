import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tonepath.cli import app
from tonepath.db import TonepathStore
from tonepath.models import Track


class CliAnalyzeTest(unittest.TestCase):
    def test_analyze_basic_stores_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                path = Path(tmp) / "song.mp3"
                path.write_bytes(b"not decoded as audio")
                track_id = store.upsert_track(
                    Track(
                        id=None,
                        path=path,
                        file_hash="hash",
                        mtime=1.0,
                        title="song",
                        artist="artist",
                        album=None,
                        genre=None,
                        duration=None,
                        format="mp3",
                    )
                )
                store.close()

                result = CliRunner().invoke(app, ["analyze", "--features", "basic"])

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Analyzed 1 track(s); skipped 0 missing track(s).", result.output)
                store = TonepathStore()
                self.assertIsNotNone(store.get_features(track_id))
                store.close()


if __name__ == "__main__":
    unittest.main()
