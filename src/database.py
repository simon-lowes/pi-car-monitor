"""
Database Module - Event Logging
================================
SQLite-based event logging for contact detection events.

Stores:
- Contact events with metadata
- Detection statistics
- Recording metadata

Privacy: No raw images or sensitive data stored.
"""

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class EventDatabase:
    """
    SQLite database for event logging.

    Thread-safe with connection pooling per thread.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Contact events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contact_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                confidence REAL,
                location_x INTEGER,
                location_y INTEGER,
                actor_type TEXT,
                actor_track_id INTEGER,
                duration REAL,
                recording_path TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Proximity events table (for statistics)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS proximity_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                object_class TEXT NOT NULL,
                track_id INTEGER,
                distance_to_car REAL,
                dwell_time REAL,
                led_to_contact INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Recordings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                duration REAL,
                trigger_type TEXT,
                file_size INTEGER,
                flagged INTEGER DEFAULT 0,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Session statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                frames_processed INTEGER DEFAULT 0,
                car_detections INTEGER DEFAULT 0,
                proximity_events INTEGER DEFAULT 0,
                contact_events INTEGER DEFAULT 0,
                recordings_triggered INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contact_timestamp
            ON contact_events(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_recordings_timestamp
            ON recordings(timestamp)
        """)

        conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    def log_contact_event(
        self,
        event_type: str,
        confidence: float,
        location: tuple = None,
        actor_type: str = None,
        actor_track_id: int = None,
        duration: float = None,
        recording_path: str = None,
        metadata: dict = None
    ) -> int:
        """
        Log a contact detection event.

        Returns:
            Event ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        import json
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO contact_events
            (timestamp, event_type, confidence, location_x, location_y,
             actor_type, actor_track_id, duration, recording_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            event_type,
            confidence,
            location[0] if location else None,
            location[1] if location else None,
            actor_type,
            actor_track_id,
            duration,
            recording_path,
            metadata_json
        ))

        conn.commit()
        event_id = cursor.lastrowid

        logger.debug(f"Logged contact event: {event_type} (id={event_id})")
        return event_id

    def log_proximity_event(
        self,
        object_class: str,
        track_id: int = None,
        distance_to_car: float = None,
        dwell_time: float = None
    ) -> int:
        """Log a proximity detection event."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO proximity_events
            (timestamp, object_class, track_id, distance_to_car, dwell_time)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            object_class,
            track_id,
            distance_to_car,
            dwell_time
        ))

        conn.commit()
        return cursor.lastrowid

    def log_recording(
        self,
        filename: str,
        duration: float = None,
        trigger_type: str = None,
        file_size: int = None
    ) -> int:
        """Log a recording."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO recordings
            (timestamp, filename, duration, trigger_type, file_size)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            filename,
            duration,
            trigger_type,
            file_size
        ))

        conn.commit()
        return cursor.lastrowid

    def start_session(self) -> int:
        """Start a new monitoring session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions (start_time)
            VALUES (?)
        """, (datetime.now().isoformat(),))

        conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id: int, stats: dict):
        """End a monitoring session with statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE sessions
            SET end_time = ?,
                frames_processed = ?,
                car_detections = ?,
                proximity_events = ?,
                contact_events = ?,
                recordings_triggered = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(),
            stats.get("frames_processed", 0),
            stats.get("car_detections", 0),
            stats.get("proximity_events", 0),
            stats.get("contact_events", 0),
            stats.get("recordings_triggered", 0),
            session_id
        ))

        conn.commit()

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent contact events."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM contact_events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def get_recent_recordings(self, limit: int = 20) -> List[Dict]:
        """Get recent recordings."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM recordings
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def get_events_by_date(self, date: str) -> List[Dict]:
        """Get events for a specific date (YYYY-MM-DD)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM contact_events
            WHERE timestamp LIKE ?
            ORDER BY timestamp ASC
        """, (f"{date}%",))

        return [dict(row) for row in cursor.fetchall()]

    def flag_recording(self, recording_id: int, flagged: bool = True, notes: str = None):
        """Flag a recording to prevent auto-deletion."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE recordings
            SET flagged = ?, notes = ?
            WHERE id = ?
        """, (1 if flagged else 0, notes, recording_id))

        conn.commit()

    def get_statistics(self, days: int = 7) -> Dict:
        """Get statistics for the last N days."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Calculate cutoff date
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Contact events count by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM contact_events
            WHERE timestamp >= ?
            GROUP BY event_type
        """, (cutoff,))
        event_counts = {row['event_type']: row['count'] for row in cursor.fetchall()}

        # Total recordings
        cursor.execute("""
            SELECT COUNT(*) as count, SUM(duration) as total_duration
            FROM recordings
            WHERE timestamp >= ?
        """, (cutoff,))
        rec_row = cursor.fetchone()

        # Proximity events
        cursor.execute("""
            SELECT object_class, COUNT(*) as count
            FROM proximity_events
            WHERE timestamp >= ?
            GROUP BY object_class
        """, (cutoff,))
        proximity_by_class = {row['object_class']: row['count'] for row in cursor.fetchall()}

        return {
            "period_days": days,
            "contact_events_by_type": event_counts,
            "total_contact_events": sum(event_counts.values()),
            "total_recordings": rec_row['count'] if rec_row else 0,
            "total_recording_duration": rec_row['total_duration'] if rec_row else 0,
            "proximity_by_class": proximity_by_class,
        }

    def cleanup_old_events(self, days: int = 90):
        """Delete events older than N days (keeps flagged recordings)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Delete old contact events
        cursor.execute("""
            DELETE FROM contact_events
            WHERE timestamp < ?
        """, (cutoff,))
        contact_deleted = cursor.rowcount

        # Delete old proximity events
        cursor.execute("""
            DELETE FROM proximity_events
            WHERE timestamp < ?
        """, (cutoff,))
        proximity_deleted = cursor.rowcount

        # Delete old unflagged recording records
        cursor.execute("""
            DELETE FROM recordings
            WHERE timestamp < ? AND flagged = 0
        """, (cutoff,))
        recordings_deleted = cursor.rowcount

        conn.commit()

        logger.info(
            f"Cleanup: deleted {contact_deleted} contact events, "
            f"{proximity_deleted} proximity events, "
            f"{recordings_deleted} recording records"
        )

        return {
            "contact_events": contact_deleted,
            "proximity_events": proximity_deleted,
            "recordings": recordings_deleted
        }

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
