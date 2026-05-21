import unittest
from pathlib import Path

from tonepath.display import canonical_track_key, dirty_metadata_issues, display_artist, display_label, display_title
from tonepath.models import Track


class DisplayTest(unittest.TestCase):
    def test_display_cleans_null_markers_and_falls_back_to_filename(self) -> None:
        track = track_for(Path("/tmp/A Song.mp3"), title="(null)", artist="Artist(null)")

        self.assertEqual(display_title(track), "A Song")
        self.assertEqual(display_artist(track), "Artist")
        self.assertEqual(display_label(track), "A Song - Artist")

    def test_dirty_metadata_marks_unknown_artist_and_missing_title(self) -> None:
        track = track_for(Path("/tmp/quiet.mp3"), title=None, artist="unknown")

        self.assertEqual(dirty_metadata_issues(track), ["dirty title", "dirty artist"])

    def test_canonical_key_groups_same_display_track_by_duration_bucket(self) -> None:
        first = track_for(Path("/tmp/one.mp3"), title="Ideal and the Real", artist="ATLUS", duration=181.0)
        second = track_for(Path("/tmp/two.mp3"), title="Ideal and the Real!", artist="atlus", duration=183.0)

        self.assertEqual(canonical_track_key(first), canonical_track_key(second))


def track_for(path: Path, title: str | None, artist: str | None, duration: float | None = 180.0) -> Track:
    """Return one display-test track."""

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


if __name__ == "__main__":
    unittest.main()
