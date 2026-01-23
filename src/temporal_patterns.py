"""
Temporal Pattern Learning
=========================
Learns and recognizes the owner's car usage patterns over time.

Features:
- Records departure/return events with timestamps
- Clusters events into recurring patterns (e.g., "Fridays 16:30-17:30")
- Matches current time against learned patterns
- Reduces alert urgency for expected departures
- Patterns decay if not reinforced

Example patterns learned:
- Weekly commute: Mon-Fri 08:00-08:30 departure, 17:00-18:00 return
- Weekend shopping: Sat 10:00-11:00
- Friday evening: Fri 16:30-17:30
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml

logger = logging.getLogger(__name__)


# Day of week names for display
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_ABBREV = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


@dataclass
class DepartureEvent:
    """Record of a single departure event."""
    timestamp: datetime
    day_of_week: int  # 0=Monday, 6=Sunday
    time_of_day: dt_time
    was_owner: bool
    owner_confidence: Optional[float] = None
    return_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    light_conditions: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'day_of_week': self.day_of_week,
            'time_of_day': self.time_of_day.strftime('%H:%M:%S'),
            'was_owner': self.was_owner,
            'owner_confidence': self.owner_confidence,
            'return_time': self.return_time.isoformat() if self.return_time else None,
            'duration_minutes': self.duration_minutes,
            'light_conditions': self.light_conditions
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DepartureEvent':
        timestamp = datetime.fromisoformat(data['timestamp'])
        time_parts = data['time_of_day'].split(':')
        time_of_day = dt_time(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]) if len(time_parts) > 2 else 0)
        return_time = datetime.fromisoformat(data['return_time']) if data.get('return_time') else None

        return cls(
            timestamp=timestamp,
            day_of_week=data['day_of_week'],
            time_of_day=time_of_day,
            was_owner=data['was_owner'],
            owner_confidence=data.get('owner_confidence'),
            return_time=return_time,
            duration_minutes=data.get('duration_minutes'),
            light_conditions=data.get('light_conditions')
        )


@dataclass
class DeparturePattern:
    """A learned pattern of owner departures."""
    pattern_id: int
    day_of_week: int  # 0=Monday, 6=Sunday
    time_window_start: dt_time
    time_window_end: dt_time
    occurrence_count: int
    last_occurred: datetime
    confidence: float  # 0-1, how reliable this pattern is

    # Optional: typical duration
    typical_duration_minutes: Optional[int] = None
    duration_std_dev: Optional[int] = None

    # Pattern metadata
    created_at: Optional[datetime] = None
    description: Optional[str] = None

    def matches_time(self, check_time: datetime) -> bool:
        """Check if a datetime falls within this pattern."""
        if check_time.weekday() != self.day_of_week:
            return False

        current_time = check_time.time()
        return self.time_window_start <= current_time <= self.time_window_end

    def to_dict(self) -> dict:
        return {
            'pattern_id': self.pattern_id,
            'day_of_week': self.day_of_week,
            'day_name': DAY_NAMES[self.day_of_week],
            'time_window_start': self.time_window_start.strftime('%H:%M'),
            'time_window_end': self.time_window_end.strftime('%H:%M'),
            'occurrence_count': self.occurrence_count,
            'last_occurred': self.last_occurred.isoformat(),
            'confidence': self.confidence,
            'typical_duration_minutes': self.typical_duration_minutes,
            'duration_std_dev': self.duration_std_dev,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DeparturePattern':
        start_parts = data['time_window_start'].split(':')
        end_parts = data['time_window_end'].split(':')

        return cls(
            pattern_id=data['pattern_id'],
            day_of_week=data['day_of_week'],
            time_window_start=dt_time(int(start_parts[0]), int(start_parts[1])),
            time_window_end=dt_time(int(end_parts[0]), int(end_parts[1])),
            occurrence_count=data['occurrence_count'],
            last_occurred=datetime.fromisoformat(data['last_occurred']),
            confidence=data['confidence'],
            typical_duration_minutes=data.get('typical_duration_minutes'),
            duration_std_dev=data.get('duration_std_dev'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            description=data.get('description')
        )

    def __str__(self) -> str:
        day = DAY_ABBREV[self.day_of_week]
        start = self.time_window_start.strftime('%H:%M')
        end = self.time_window_end.strftime('%H:%M')
        return f"{day} {start}-{end} ({self.occurrence_count}x, {self.confidence:.0%})"


class TemporalPatternDB:
    """
    Database for learning and matching temporal patterns.

    Learns from confirmed owner departures and returns.
    """

    def __init__(
        self,
        data_path: str = "data/temporal_patterns.yaml",
        config: dict = None
    ):
        self.data_path = Path(data_path)
        self.config = config or {}

        # Configuration
        self.min_occurrences = self.config.get('min_occurrences_for_pattern', 2)
        self.time_cluster_window_minutes = self.config.get('time_cluster_window_minutes', 30)
        self.confidence_base = self.config.get('pattern_confidence_base', 0.5)
        self.confidence_per_occurrence = self.config.get('pattern_confidence_per_occurrence', 0.1)
        self.max_confidence = self.config.get('max_pattern_confidence', 0.95)
        self.decay_days = self.config.get('pattern_decay_days', 30)  # Days without occurrence before decay
        self.max_events = self.config.get('max_stored_events', 500)

        # Data
        self.patterns: List[DeparturePattern] = []
        self.events: List[DepartureEvent] = []
        self._next_pattern_id = 1

        # Pending departure (waiting for return)
        self._pending_departure: Optional[DepartureEvent] = None

        # Load existing data
        self._load()

        logger.info(f"TemporalPatternDB initialized: {len(self.patterns)} patterns, {len(self.events)} events")

    def _load(self):
        """Load patterns and events from file."""
        if not self.data_path.exists():
            return

        try:
            with open(self.data_path, 'r') as f:
                data = yaml.safe_load(f)

            if data:
                # Load patterns
                for p_data in data.get('patterns', []):
                    try:
                        pattern = DeparturePattern.from_dict(p_data)
                        self.patterns.append(pattern)
                        if pattern.pattern_id >= self._next_pattern_id:
                            self._next_pattern_id = pattern.pattern_id + 1
                    except Exception as e:
                        logger.warning(f"Failed to load pattern: {e}")

                # Load events
                for e_data in data.get('events', []):
                    try:
                        event = DepartureEvent.from_dict(e_data)
                        self.events.append(event)
                    except Exception as e:
                        logger.warning(f"Failed to load event: {e}")

                logger.info(f"Loaded {len(self.patterns)} patterns and {len(self.events)} events")

        except Exception as e:
            logger.error(f"Failed to load temporal patterns: {e}")

    def _save(self):
        """Save patterns and events to file."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'patterns': [p.to_dict() for p in self.patterns],
                'events': [e.to_dict() for e in self.events[-self.max_events:]],
                'last_updated': datetime.now().isoformat(),
                'version': 1
            }

            with open(self.data_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            logger.debug("Temporal patterns saved")

        except Exception as e:
            logger.error(f"Failed to save temporal patterns: {e}")

    def record_departure(
        self,
        timestamp: datetime,
        was_owner: bool,
        owner_confidence: Optional[float] = None,
        light_conditions: Optional[str] = None
    ) -> DepartureEvent:
        """
        Record a departure event.

        Args:
            timestamp: When the departure occurred
            was_owner: Whether this was confirmed as the owner
            owner_confidence: Recognition confidence if available
            light_conditions: Light conditions at time of departure

        Returns:
            The created DepartureEvent
        """
        event = DepartureEvent(
            timestamp=timestamp,
            day_of_week=timestamp.weekday(),
            time_of_day=timestamp.time(),
            was_owner=was_owner,
            owner_confidence=owner_confidence,
            light_conditions=light_conditions
        )

        self.events.append(event)
        self._pending_departure = event

        # Trim old events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Update patterns if this was the owner
        if was_owner:
            self._update_patterns()

        self._save()

        logger.info(f"Departure recorded: {DAY_ABBREV[event.day_of_week]} {event.time_of_day.strftime('%H:%M')}, owner={was_owner}")

        return event

    def record_return(self, return_time: datetime) -> Optional[int]:
        """
        Record a return event and update the pending departure.

        Args:
            return_time: When the car returned

        Returns:
            Duration in minutes, or None if no pending departure
        """
        if not self._pending_departure:
            logger.debug("No pending departure to update with return time")
            return None

        departure = self._pending_departure
        departure.return_time = return_time
        duration = int((return_time - departure.timestamp).total_seconds() / 60)
        departure.duration_minutes = duration

        self._pending_departure = None

        # Update pattern duration stats
        self._update_duration_stats(departure)

        self._save()

        logger.info(f"Return recorded: duration={duration} minutes")

        return duration

    def _update_patterns(self):
        """Analyze events and update patterns."""
        # Get only owner-confirmed events
        owner_events = [e for e in self.events if e.was_owner]

        if len(owner_events) < self.min_occurrences:
            return

        # Group by day of week
        by_day: Dict[int, List[DepartureEvent]] = defaultdict(list)
        for event in owner_events:
            by_day[event.day_of_week].append(event)

        # Find time clusters for each day
        new_patterns = []
        for day, day_events in by_day.items():
            clusters = self._cluster_times(day_events)

            for cluster in clusters:
                if len(cluster['events']) >= self.min_occurrences:
                    # Calculate confidence
                    count = len(cluster['events'])
                    confidence = min(
                        self.max_confidence,
                        self.confidence_base + (count * self.confidence_per_occurrence)
                    )

                    # Find existing pattern or create new
                    existing = self._find_matching_pattern(day, cluster['start'], cluster['end'])

                    if existing:
                        # Update existing pattern
                        existing.occurrence_count = count
                        existing.time_window_start = cluster['start']
                        existing.time_window_end = cluster['end']
                        existing.last_occurred = max(e.timestamp for e in cluster['events'])
                        existing.confidence = confidence
                    else:
                        # Create new pattern
                        pattern = DeparturePattern(
                            pattern_id=self._next_pattern_id,
                            day_of_week=day,
                            time_window_start=cluster['start'],
                            time_window_end=cluster['end'],
                            occurrence_count=count,
                            last_occurred=max(e.timestamp for e in cluster['events']),
                            confidence=confidence,
                            created_at=datetime.now()
                        )
                        self._next_pattern_id += 1
                        new_patterns.append(pattern)

        # Add new patterns
        self.patterns.extend(new_patterns)

        # Apply decay to old patterns
        self._apply_pattern_decay()

        # Remove patterns with very low confidence
        self.patterns = [p for p in self.patterns if p.confidence > 0.2]

        if new_patterns:
            logger.info(f"Created {len(new_patterns)} new patterns")

    def _cluster_times(
        self,
        events: List[DepartureEvent]
    ) -> List[Dict[str, Any]]:
        """
        Cluster events by time of day.

        Returns list of clusters with start, end times and events.
        """
        if not events:
            return []

        # Sort by time
        sorted_events = sorted(events, key=lambda e: e.time_of_day)

        clusters = []
        current_cluster = {
            'events': [sorted_events[0]],
            'start': sorted_events[0].time_of_day,
            'end': sorted_events[0].time_of_day
        }

        window_delta = timedelta(minutes=self.time_cluster_window_minutes)

        for event in sorted_events[1:]:
            event_time = event.time_of_day
            cluster_end = current_cluster['end']

            # Convert to datetime for comparison
            today = datetime.now().date()
            event_dt = datetime.combine(today, event_time)
            cluster_end_dt = datetime.combine(today, cluster_end)

            if event_dt - cluster_end_dt <= window_delta:
                # Add to current cluster
                current_cluster['events'].append(event)
                current_cluster['end'] = event_time
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = {
                    'events': [event],
                    'start': event_time,
                    'end': event_time
                }

        clusters.append(current_cluster)

        # Expand time windows slightly for matching
        for cluster in clusters:
            start = cluster['start']
            end = cluster['end']

            # Add padding (15 minutes on each side)
            start_dt = datetime.combine(today, start) - timedelta(minutes=15)
            end_dt = datetime.combine(today, end) + timedelta(minutes=15)

            # Handle day boundaries
            cluster['start'] = max(dt_time(0, 0), start_dt.time())
            cluster['end'] = min(dt_time(23, 59), end_dt.time())

        return clusters

    def _find_matching_pattern(
        self,
        day: int,
        start: dt_time,
        end: dt_time
    ) -> Optional[DeparturePattern]:
        """Find an existing pattern that overlaps with the given time window."""
        for pattern in self.patterns:
            if pattern.day_of_week != day:
                continue

            # Check for overlap
            if (pattern.time_window_start <= end and
                pattern.time_window_end >= start):
                return pattern

        return None

    def _apply_pattern_decay(self):
        """Apply confidence decay to patterns that haven't occurred recently."""
        now = datetime.now()
        decay_threshold = now - timedelta(days=self.decay_days)

        for pattern in self.patterns:
            if pattern.last_occurred < decay_threshold:
                days_since = (now - pattern.last_occurred).days
                decay_factor = 1.0 - ((days_since - self.decay_days) * 0.01)
                decay_factor = max(0.0, decay_factor)

                pattern.confidence *= decay_factor

                if pattern.confidence < 0.3:
                    logger.info(f"Pattern {pattern} decaying (confidence: {pattern.confidence:.2f})")

    def _update_duration_stats(self, event: DepartureEvent):
        """Update duration statistics for matching patterns."""
        if event.duration_minutes is None:
            return

        matching_pattern = self.is_expected_departure(event.timestamp)[1]
        if not matching_pattern:
            return

        # Get all durations for this pattern
        durations = []
        for e in self.events:
            if e.duration_minutes and matching_pattern.matches_time(e.timestamp):
                durations.append(e.duration_minutes)

        if durations:
            import statistics
            matching_pattern.typical_duration_minutes = int(statistics.mean(durations))
            if len(durations) >= 2:
                matching_pattern.duration_std_dev = int(statistics.stdev(durations))

    def is_expected_departure(
        self,
        timestamp: datetime
    ) -> Tuple[bool, Optional[DeparturePattern]]:
        """
        Check if a departure at this time matches a known pattern.

        Args:
            timestamp: Time to check

        Returns:
            (is_expected, matching_pattern)
        """
        for pattern in self.patterns:
            if pattern.matches_time(timestamp):
                return True, pattern

        return False, None

    def get_patterns_for_day(self, day_of_week: int) -> List[DeparturePattern]:
        """Get all patterns for a specific day."""
        return [p for p in self.patterns if p.day_of_week == day_of_week]

    def get_all_patterns(self) -> List[DeparturePattern]:
        """Get all patterns sorted by day and time."""
        return sorted(
            self.patterns,
            key=lambda p: (p.day_of_week, p.time_window_start)
        )

    def get_pattern_summary(self) -> str:
        """Get a human-readable summary of learned patterns."""
        if not self.patterns:
            return "No patterns learned yet"

        lines = ["Learned usage patterns:"]
        for day in range(7):
            day_patterns = self.get_patterns_for_day(day)
            if day_patterns:
                for p in sorted(day_patterns, key=lambda x: x.time_window_start):
                    start = p.time_window_start.strftime('%H:%M')
                    end = p.time_window_end.strftime('%H:%M')
                    duration = f", ~{p.typical_duration_minutes}min" if p.typical_duration_minutes else ""
                    lines.append(f"  {DAY_NAMES[day]}: {start}-{end} ({p.occurrence_count}x{duration})")

        return "\n".join(lines)

    def get_recent_events(self, days: int = 7) -> List[DepartureEvent]:
        """Get events from the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in self.events if e.timestamp > cutoff]

    def get_status(self) -> Dict[str, Any]:
        """Get database status for debugging."""
        return {
            'pattern_count': len(self.patterns),
            'event_count': len(self.events),
            'has_pending_departure': self._pending_departure is not None,
            'patterns': [str(p) for p in self.patterns],
            'recent_events': len(self.get_recent_events(7))
        }


def create_pattern_db_from_config(config: dict) -> Optional[TemporalPatternDB]:
    """Create TemporalPatternDB from configuration."""
    temporal_config = config.get('temporal_patterns', {})

    if not temporal_config.get('enabled', True):
        logger.info("Temporal pattern learning disabled")
        return None

    data_path = temporal_config.get(
        'data_path',
        'data/temporal_patterns.yaml'
    )

    return TemporalPatternDB(data_path=data_path, config=temporal_config)
