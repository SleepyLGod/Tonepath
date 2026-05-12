import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tonepath.db import TonepathStore
from tonepath.models import Track
from tonepath.tui import TonepathApp


class TonepathTuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_tui_launches_session_screen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"TONEPATH_HOME": str(Path(tmp) / "home")}):
                store = TonepathStore()
                self.add_track(store, tmp, "a.mp3")
                self.add_track(store, tmp, "b.mp3")
                store.close()

                app = TonepathApp("from irritated to focus in 30 minutes")
                async with app.run_test() as pilot:
                    self.assertIsNotNone(app.query_one("#timeline"))
                    self.assertIsNotNone(app.query_one("#queue"))
                    self.assertIsNotNone(app.query_one("#why-panel"))
                    self.assertIsNotNone(app.query_one("#event-log"))
                    await pilot.press("w")
                    await pilot.press("s")

    def add_track(self, store: TonepathStore, tmp: str, name: str) -> int:
        path = Path(tmp) / name
        path.write_bytes(b"not real audio")
        return store.upsert_track(
            Track(
                id=None,
                path=path,
                file_hash=name,
                mtime=1.0,
                title=name,
                artist="artist",
                album=None,
                genre=None,
                duration=None,
                format="mp3",
            )
        )


if __name__ == "__main__":
    unittest.main()
