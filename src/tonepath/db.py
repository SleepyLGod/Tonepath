"""SQLite persistence for Tonepath."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from tonepath import config
from tonepath.models import EnrichmentRecord, SessionPhase, SessionPlan, Track, TrackFeatures


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL UNIQUE,
  file_hash TEXT NOT NULL,
  mtime REAL NOT NULL,
  title TEXT,
  artist TEXT,
  album TEXT,
  genre TEXT,
  duration REAL,
  format TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS track_features (
  track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
  bpm REAL,
  loudness REAL,
  energy REAL,
  vocalness REAL,
  arousal_estimate REAL,
  valence_estimate REAL,
  feature_source TEXT NOT NULL,
  confidence TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  prompt TEXT NOT NULL,
  source_state TEXT NOT NULL,
  target_state TEXT NOT NULL,
  duration_sec INTEGER NOT NULL,
  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ended_at TEXT,
  network_mode TEXT NOT NULL DEFAULT 'offline'
);

CREATE TABLE IF NOT EXISTS session_phases (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  label TEXT NOT NULL,
  start_sec INTEGER NOT NULL,
  end_sec INTEGER NOT NULL,
  target_arousal REAL NOT NULL,
  target_valence REAL NOT NULL,
  target_energy REAL NOT NULL,
  vocal_policy TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plays (
  id INTEGER PRIMARY KEY,
  session_id INTEGER,
  track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  phase_id INTEGER,
  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ended_at TEXT,
  skipped INTEGER NOT NULL DEFAULT 0,
  position_sec REAL
);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY,
  session_id INTEGER,
  track_id INTEGER,
  type TEXT NOT NULL,
  value TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS profile_rules (
  id INTEGER PRIMARY KEY,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS track_enrichment (
  id INTEGER PRIMARY KEY,
  track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  field TEXT NOT NULL,
  value TEXT NOT NULL,
  tier TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence TEXT NOT NULL,
  is_online INTEGER NOT NULL DEFAULT 0,
  fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(track_id, field, tier, source)
);

CREATE TABLE IF NOT EXISTS app_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


class TonepathStore:
    """Small SQLite repository for Tonepath local state."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        """Close the database connection."""

        self.conn.close()

    def init_schema(self) -> None:
        """Create required tables if they do not exist."""

        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_track(self, track: Track) -> int:
        """Insert or update a local track and return its database id."""

        self.conn.execute(
            """
            INSERT INTO tracks (
              path, file_hash, mtime, title, artist, album, genre, duration, format
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              file_hash=excluded.file_hash,
              mtime=excluded.mtime,
              title=excluded.title,
              artist=excluded.artist,
              album=excluded.album,
              genre=excluded.genre,
              duration=excluded.duration,
              format=excluded.format,
              updated_at=CURRENT_TIMESTAMP
            """,
            (
                str(track.path),
                track.file_hash,
                track.mtime,
                track.title,
                track.artist,
                track.album,
                track.genre,
                track.duration,
                track.format,
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM tracks WHERE path = ?", (str(track.path),)).fetchone()
        if row is None:
            raise RuntimeError("Track upsert did not return a row.")
        return int(row["id"])

    def list_tracks(self) -> list[Track]:
        """Return all known tracks."""

        rows = self.conn.execute("SELECT * FROM tracks ORDER BY artist, album, title, path").fetchall()
        return [track_from_row(row) for row in rows]

    def get_track(self, track_id: int) -> Track | None:
        """Return one known track by id."""

        row = self.conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
        return track_from_row(row) if row else None

    def get_features(self, track_id: int) -> TrackFeatures | None:
        """Return locally analyzed features for one track."""

        row = self.conn.execute("SELECT * FROM track_features WHERE track_id = ?", (track_id,)).fetchone()
        if row is None:
            return None
        return TrackFeatures(
            track_id=int(row["track_id"]),
            bpm=row["bpm"],
            loudness=row["loudness"],
            energy=row["energy"],
            vocalness=row["vocalness"],
            arousal_estimate=row["arousal_estimate"],
            valence_estimate=row["valence_estimate"],
            feature_source=row["feature_source"],
            confidence=row["confidence"],
        )

    def upsert_features(self, features: TrackFeatures) -> None:
        """Insert or update locally analyzed features for one track."""

        self.conn.execute(
            """
            INSERT INTO track_features (
              track_id, bpm, loudness, energy, vocalness, arousal_estimate,
              valence_estimate, feature_source, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
              bpm=excluded.bpm,
              loudness=excluded.loudness,
              energy=excluded.energy,
              vocalness=excluded.vocalness,
              arousal_estimate=excluded.arousal_estimate,
              valence_estimate=excluded.valence_estimate,
              feature_source=excluded.feature_source,
              confidence=excluded.confidence
            """,
            (
                features.track_id,
                features.bpm,
                features.loudness,
                features.energy,
                features.vocalness,
                features.arousal_estimate,
                features.valence_estimate,
                features.feature_source,
                features.confidence,
            ),
        )
        self.conn.commit()

    def save_session(self, plan: SessionPlan) -> int:
        """Persist a session and its phases."""

        cursor = self.conn.execute(
            """
            INSERT INTO sessions (prompt, source_state, target_state, duration_sec)
            VALUES (?, ?, ?, ?)
            """,
            (
                plan.request.prompt,
                plan.request.source_state,
                plan.request.target_state,
                plan.request.duration_sec,
            ),
        )
        session_id = int(cursor.lastrowid)
        self.save_phases(session_id, plan.phases)
        self.conn.execute(
            "INSERT OR REPLACE INTO app_state (key, value) VALUES ('current_session_id', ?)",
            (str(session_id),),
        )
        self.conn.commit()
        return session_id

    def save_phases(self, session_id: int, phases: Iterable[SessionPhase]) -> None:
        """Persist phases for a session."""

        self.conn.executemany(
            """
            INSERT INTO session_phases (
              session_id, label, start_sec, end_sec, target_arousal,
              target_valence, target_energy, vocal_policy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    phase.label,
                    phase.start_sec,
                    phase.end_sec,
                    phase.target_arousal,
                    phase.target_valence,
                    phase.target_energy,
                    phase.vocal_policy,
                )
                for phase in phases
            ],
        )

    def current_session_id(self) -> int | None:
        """Return the current session id if one is known."""

        value = self.get_app_state("current_session_id")
        return int(value) if value else None

    def set_app_state(self, key: str, value: str) -> None:
        """Set one local application state value."""

        self.conn.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get_app_state(self, key: str) -> str | None:
        """Return one local application state value."""

        row = self.conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def delete_app_state(self, key: str) -> None:
        """Delete one local application state value."""

        self.conn.execute("DELETE FROM app_state WHERE key = ?", (key,))
        self.conn.commit()

    def record_feedback(
        self,
        feedback_type: str,
        value: str | None = None,
        session_id: int | None = None,
        track_id: int | None = None,
    ) -> None:
        """Record local user feedback."""

        self.conn.execute(
            "INSERT INTO feedback (session_id, track_id, type, value) VALUES (?, ?, ?, ?)",
            (session_id, track_id, feedback_type, value),
        )
        self.conn.commit()

    def start_play(self, session_id: int | None, track_id: int) -> int:
        """Record the start of one local playback event."""

        cursor = self.conn.execute(
            "INSERT INTO plays (session_id, track_id) VALUES (?, ?)",
            (session_id, track_id),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def end_play(self, play_id: int, skipped: bool = False) -> None:
        """Mark one local playback event as ended."""

        self.conn.execute(
            """
            UPDATE plays
            SET ended_at = CURRENT_TIMESTAMP, skipped = ?
            WHERE id = ?
            """,
            (1 if skipped else 0, play_id),
        )
        self.conn.commit()

    def feedback_counts_for_track(self, track_id: int) -> dict[str, int]:
        """Return feedback counts for one track."""

        rows = self.conn.execute(
            "SELECT type, COUNT(*) AS count FROM feedback WHERE track_id = ? GROUP BY type",
            (track_id,),
        ).fetchall()
        return {str(row["type"]): int(row["count"]) for row in rows}

    def profile_summary(self) -> dict[str, int]:
        """Return counts of locally stored user data."""

        summary: dict[str, int] = {}
        for table in (
            "tracks",
            "track_features",
            "track_enrichment",
            "sessions",
            "session_phases",
            "plays",
            "feedback",
            "profile_rules",
        ):
            row = self.conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
            summary[table] = int(row["count"])
        return summary

    def upsert_enrichment(self, record: EnrichmentRecord) -> None:
        """Insert or update a source-attributed track enrichment field."""

        self.conn.execute(
            """
            INSERT INTO track_enrichment (
              track_id, field, value, tier, source, confidence, is_online
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id, field, tier, source) DO UPDATE SET
              value=excluded.value,
              confidence=excluded.confidence,
              is_online=excluded.is_online,
              fetched_at=CURRENT_TIMESTAMP
            """,
            (
                record.track_id,
                record.field,
                record.value,
                record.tier,
                record.source,
                record.confidence,
                1 if record.is_online else 0,
            ),
        )
        self.conn.commit()

    def list_enrichment(self, track_id: int) -> list[EnrichmentRecord]:
        """Return source-attributed enrichment fields for one track."""

        rows = self.conn.execute(
            """
            SELECT track_id, field, value, tier, source, confidence, is_online
            FROM track_enrichment
            WHERE track_id = ?
            ORDER BY tier, field, source
            """,
            (track_id,),
        ).fetchall()
        return [
            EnrichmentRecord(
                track_id=int(row["track_id"]),
                field=str(row["field"]),
                value=str(row["value"]),
                tier=row["tier"],
                source=str(row["source"]),
                confidence=str(row["confidence"]),
                is_online=bool(row["is_online"]),
            )
            for row in rows
        ]

    def delete_profile_data(self) -> None:
        """Delete user profile, play, feedback, and session data while keeping scanned tracks."""

        for table in ("profile_rules", "feedback", "plays", "session_phases", "sessions"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.execute("DELETE FROM app_state WHERE key = 'current_session_id'")
        self.conn.commit()


def track_from_row(row: sqlite3.Row) -> Track:
    """Build a Track from a SQLite row."""

    return Track(
        id=int(row["id"]),
        path=Path(row["path"]),
        file_hash=row["file_hash"],
        mtime=float(row["mtime"]),
        title=row["title"],
        artist=row["artist"],
        album=row["album"],
        genre=row["genre"],
        duration=row["duration"],
        format=row["format"],
    )
