import tempfile
import unittest
from pathlib import Path

from tonepath.scanner import scan_directory


class ScannerTest(unittest.TestCase):
    def test_scans_chinese_path_with_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "音乐"
            root.mkdir()
            path = root / "测试歌曲.mp3"
            path.write_bytes(b"not real audio but acceptable for fallback")
            tracks = scan_directory(root)
            self.assertEqual(len(tracks), 1)
            self.assertEqual(tracks[0].title, "测试歌曲")
            self.assertEqual(tracks[0].format, "mp3")


if __name__ == "__main__":
    unittest.main()

