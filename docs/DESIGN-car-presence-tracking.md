# Design: Car Presence Tracking & Departure Detection

## Overview

This document outlines the design for extending pi-car-monitor with:
1. Dynamic car position tracking
2. Departure/return detection (including low-light)
3. Owner identification at departure
4. Temporal pattern learning

## Goals

- Detect when the car leaves its parking position (even in low light)
- Alert the owner and allow "it was me" confirmation
- Learn the owner's appearance and typical usage patterns
- Reduce false alerts over time through pattern recognition
- Re-establish car position when it returns

---

## 1. Car Presence State Machine

### States

```
                    ┌─────────────┐
                    │   UNKNOWN   │ (startup, no baseline)
                    └──────┬──────┘
                           │ car detected in zone
                           ▼
    ┌──────────────────────────────────────────────────┐
    │                    PRESENT                        │
    │  (car in known position, baseline established)    │
    └───────────┬──────────────────────────┬───────────┘
                │                          │
                │ departure signals        │ contact detected
                ▼                          ▼
    ┌───────────────────┐        ┌─────────────────────┐
    │    DEPARTING      │        │  (existing pipeline) │
    │ (motion in zone,  │        │  contact alerts      │
    │  lights detected) │        └─────────────────────┘
    └─────────┬─────────┘
              │ car no longer in zone
              ▼
    ┌───────────────────┐
    │     ABSENT        │
    │ (car not detected │
    │  for N frames)    │
    └─────────┬─────────┘
              │ car detected in zone
              ▼
    ┌───────────────────┐
    │    RETURNING      │
    │ (car re-entering, │
    │  position update) │
    └─────────┬─────────┘
              │ stable detection
              ▼
         Back to PRESENT
```

### State Transitions

| From | To | Trigger |
|------|-----|---------|
| UNKNOWN | PRESENT | Car detected with high confidence, daylight, stable for 10+ frames |
| PRESENT | DEPARTING | Motion in car zone OR tail lights detected OR car bbox shrinking/moving |
| DEPARTING | ABSENT | Car not detected for 30+ consecutive frames |
| DEPARTING | PRESENT | Motion stops, car still in position (false alarm) |
| ABSENT | RETURNING | Car detected entering zone |
| RETURNING | PRESENT | Car stable in zone for 10+ frames, position baseline updated |

### New File: `src/presence_tracker.py`

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple
import time

class PresenceState(Enum):
    UNKNOWN = auto()
    PRESENT = auto()
    DEPARTING = auto()
    ABSENT = auto()
    RETURNING = auto()

@dataclass
class CarBaseline:
    """Established car position and appearance baseline."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    center: Tuple[int, int]
    area: int
    established_at: float
    light_conditions: str  # 'daylight', 'low_light', 'night'
    confidence: float

    # Tolerance for position drift (pixels)
    position_tolerance: int = 50
    area_tolerance_pct: float = 0.15  # 15% size change allowed

@dataclass
class PresenceTracker:
    state: PresenceState = PresenceState.UNKNOWN
    baseline: Optional[CarBaseline] = None

    # Counters for state stability
    frames_in_state: int = 0
    frames_car_missing: int = 0
    frames_car_detected: int = 0

    # Departure detection
    departure_started_at: Optional[float] = None
    departure_trigger: Optional[str] = None  # 'motion', 'lights', 'position_shift'

    # For learning
    last_departure_time: Optional[float] = None
    last_return_time: Optional[float] = None
```

---

## 2. Dynamic Car Position Tracking

### Baseline Establishment

The system should establish a car position baseline when:
1. **First startup** - Car detected with high confidence in daylight
2. **After return** - Car re-enters zone and stabilizes
3. **Manual calibration** - User triggers recalibration

### Baseline Data Stored

```yaml
# data/car_baseline.yaml
baseline:
  bbox:
    x1: 245
    y1: 180
    x2: 520
    y2: 410
  center: [382, 295]
  area: 63250
  established_at: "2025-01-23T14:30:00"
  light_conditions: "daylight"
  frame_snapshot: "data/baselines/baseline_20250123_143000.jpg"

  # Acceptable variance
  position_tolerance_px: 50
  area_tolerance_pct: 0.15

history:
  - established_at: "2025-01-22T09:15:00"
    center: [380, 293]
    area: 62800
  - established_at: "2025-01-21T17:45:00"
    center: [385, 298]
    area: 63500
```

### Position Comparison Logic

```python
def is_car_in_baseline_position(current_bbox, baseline: CarBaseline) -> Tuple[bool, float]:
    """
    Check if car is in its expected position.
    Returns (is_in_position, confidence)
    """
    current_center = get_center(current_bbox)
    current_area = get_area(current_bbox)

    # Distance from baseline center
    distance = euclidean_distance(current_center, baseline.center)

    # Area ratio
    area_ratio = current_area / baseline.area

    # Position score (1.0 = perfect match, 0.0 = outside tolerance)
    position_score = max(0, 1 - (distance / baseline.position_tolerance))

    # Area score
    area_deviation = abs(1 - area_ratio)
    area_score = max(0, 1 - (area_deviation / baseline.area_tolerance_pct))

    # Combined confidence
    confidence = (position_score * 0.6) + (area_score * 0.4)

    in_position = distance <= baseline.position_tolerance and \
                  (1 - baseline.area_tolerance_pct) <= area_ratio <= (1 + baseline.area_tolerance_pct)

    return in_position, confidence
```

---

## 3. Low-Light Departure Detection

### Challenge

In darkness, the car itself may not be detectable via standard object detection, but departure signals are visible:
- **Tail lights** illuminating (red glow in car zone)
- **Headlights** turning on (bright area at front)
- **Reverse lights** (white light at rear)
- **Interior lights** (dome light when door opens)
- **Motion blur** in known car zone

### Detection Methods

#### 3.1 Light Detection in Car Zone

```python
@dataclass
class LightEvent:
    type: str  # 'tail_light', 'headlight', 'reverse_light', 'interior'
    location: Tuple[int, int]  # center of light blob
    intensity: float  # 0-1
    color: str  # 'red', 'white', 'amber'
    timestamp: float

def detect_lights_in_zone(frame, car_zone_bbox) -> List[LightEvent]:
    """
    Detect illumination events within the known car zone.
    Works in low-light conditions where object detection fails.
    """
    # Extract car zone region
    x1, y1, x2, y2 = car_zone_bbox
    zone = frame[y1:y2, x1:x2]

    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

    events = []

    # Tail lights: Red, high saturation, high value
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
               cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))

    # Reverse/headlights: White/bright, low saturation, high value
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))

    # Find contours and analyze
    for mask, light_type, color in [
        (red_mask, 'tail_light', 'red'),
        (white_mask, 'reverse_light', 'white')
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum blob size
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    intensity = area / (zone.shape[0] * zone.shape[1])
                    events.append(LightEvent(
                        type=light_type,
                        location=(cx, cy),
                        intensity=min(1.0, intensity * 10),
                        color=color,
                        timestamp=time.time()
                    ))

    return events
```

#### 3.2 Motion Detection in Car Zone

Even without identifying the car, detect motion where it should be:

```python
def detect_zone_motion(prev_frame, curr_frame, car_zone_bbox, threshold=25) -> dict:
    """
    Detect motion within the car zone using frame differencing.
    """
    x1, y1, x2, y2 = car_zone_bbox

    prev_zone = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    curr_zone = cv2.cvtColor(curr_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    prev_zone = cv2.GaussianBlur(prev_zone, (21, 21), 0)
    curr_zone = cv2.GaussianBlur(curr_zone, (21, 21), 0)

    # Frame difference
    diff = cv2.absdiff(prev_zone, curr_zone)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Calculate motion metrics
    motion_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    motion_ratio = motion_pixels / total_pixels

    # Find motion centroid
    M = cv2.moments(thresh)
    motion_center = None
    if M["m00"] > 0:
        motion_center = (
            int(M["m10"] / M["m00"]) + x1,
            int(M["m01"] / M["m00"]) + y1
        )

    return {
        'motion_detected': motion_ratio > 0.02,  # 2% of zone changed
        'motion_ratio': motion_ratio,
        'motion_center': motion_center,
        'is_significant': motion_ratio > 0.10  # 10% = major movement
    }
```

#### 3.3 Combined Departure Signal Detection

```python
@dataclass
class DepartureSignal:
    timestamp: float
    signal_type: str  # 'lights', 'motion', 'position_shift', 'car_missing'
    confidence: float
    details: dict

def detect_departure_signals(
    frame,
    prev_frame,
    car_detection,  # May be None in low light
    baseline: CarBaseline,
    light_conditions: str
) -> List[DepartureSignal]:
    """
    Detect signals that suggest the car is departing.
    Works in both daylight and low-light conditions.
    """
    signals = []
    car_zone = baseline.bbox

    # Method 1: Light detection (works in dark)
    lights = detect_lights_in_zone(frame, car_zone)
    if lights:
        tail_lights = [l for l in lights if l.type == 'tail_light']
        reverse_lights = [l for l in lights if l.type == 'reverse_light']

        if tail_lights:
            signals.append(DepartureSignal(
                timestamp=time.time(),
                signal_type='tail_lights',
                confidence=0.7,
                details={'count': len(tail_lights), 'lights': tail_lights}
            ))

        if reverse_lights:
            signals.append(DepartureSignal(
                timestamp=time.time(),
                signal_type='reverse_lights',
                confidence=0.85,  # Strong indicator of imminent departure
                details={'count': len(reverse_lights), 'lights': reverse_lights}
            ))

    # Method 2: Motion in car zone (works in any light)
    if prev_frame is not None:
        motion = detect_zone_motion(prev_frame, frame, car_zone)
        if motion['is_significant']:
            signals.append(DepartureSignal(
                timestamp=time.time(),
                signal_type='zone_motion',
                confidence=0.6,
                details=motion
            ))

    # Method 3: Car position shift (daylight only, needs detection)
    if car_detection is not None:
        in_position, pos_confidence = is_car_in_baseline_position(
            car_detection.bbox, baseline
        )
        if not in_position:
            signals.append(DepartureSignal(
                timestamp=time.time(),
                signal_type='position_shift',
                confidence=0.8,
                details={
                    'expected': baseline.center,
                    'actual': get_center(car_detection.bbox),
                    'position_confidence': pos_confidence
                }
            ))

    # Method 4: Car no longer detected (daylight only)
    if light_conditions == 'daylight' and car_detection is None:
        signals.append(DepartureSignal(
            timestamp=time.time(),
            signal_type='car_missing',
            confidence=0.9,
            details={'light_conditions': light_conditions}
        ))

    return signals
```

---

## 4. Owner Identification at Departure

### Current Limitation

The existing `owner_profile.py` only trains from "contact" events - when someone is near the car. For departure detection, we need to identify the owner **getting into** the car.

### Enhanced Owner Detection

#### 4.1 Capture Owner at Departure

When departure signals are detected, capture and analyze any person in/near the car:

```python
def capture_departure_actor(frame, car_zone, pose_detections) -> Optional[dict]:
    """
    Identify and capture features of person involved in departure.
    """
    # Find people overlapping with car zone
    car_x1, car_y1, car_x2, car_y2 = car_zone

    for pose in pose_detections:
        person_bbox = pose['bbox']

        # Check if person overlaps with car (getting in)
        overlap = calculate_iou(person_bbox, car_zone)

        if overlap > 0.1:  # At least 10% overlap
            return {
                'bbox': person_bbox,
                'pose_keypoints': pose.get('keypoints'),
                'overlap_ratio': overlap,
                'position': 'entering_car' if overlap > 0.3 else 'near_car',
                'frame_crop': extract_person_crop(frame, person_bbox),
                'timestamp': time.time()
            }

    return None
```

#### 4.2 Match Against Owner Profile

```python
def is_likely_owner(actor_data: dict, owner_profile: OwnerProfile) -> Tuple[bool, float]:
    """
    Determine if the departing person matches the owner profile.
    """
    if not owner_profile.is_trained():
        return False, 0.0

    # Extract features from departure actor
    features = extract_appearance_features(actor_data['frame_crop'])

    # Compare against stored owner samples
    similarity = owner_profile.match(features)

    # Threshold for owner match
    is_owner = similarity >= 0.7

    return is_owner, similarity
```

---

## 5. Temporal Pattern Learning

### Data Model

```python
@dataclass
class DeparturePattern:
    """Learned pattern of owner departures."""
    day_of_week: int  # 0=Monday, 6=Sunday
    time_window_start: time  # e.g., 16:30
    time_window_end: time    # e.g., 17:30
    occurrence_count: int
    last_occurred: datetime
    confidence: float  # How reliable this pattern is

    # Optional: duration patterns
    typical_duration_minutes: Optional[int] = None
    duration_std_dev: Optional[int] = None

@dataclass
class TemporalPatternDB:
    patterns: List[DeparturePattern]
    departures: List[dict]  # Raw departure events for learning

    def record_departure(self, timestamp: datetime, was_owner: bool, duration: Optional[int]):
        """Record a departure event for pattern learning."""
        self.departures.append({
            'timestamp': timestamp,
            'day_of_week': timestamp.weekday(),
            'time': timestamp.time(),
            'was_owner': was_owner,
            'duration_minutes': duration
        })
        self._update_patterns()

    def _update_patterns(self):
        """Analyze departures and update patterns."""
        # Group by day of week
        by_day = defaultdict(list)
        for dep in self.departures:
            if dep['was_owner']:
                by_day[dep['day_of_week']].append(dep)

        # Find time clusters for each day
        self.patterns = []
        for day, events in by_day.items():
            clusters = self._cluster_times([e['time'] for e in events])
            for cluster in clusters:
                if len(cluster['times']) >= 2:  # Need at least 2 occurrences
                    self.patterns.append(DeparturePattern(
                        day_of_week=day,
                        time_window_start=cluster['start'],
                        time_window_end=cluster['end'],
                        occurrence_count=len(cluster['times']),
                        last_occurred=max(e['timestamp'] for e in events
                                         if cluster['start'] <= e['time'] <= cluster['end']),
                        confidence=min(0.95, 0.5 + (len(cluster['times']) * 0.1))
                    ))

    def is_expected_departure(self, timestamp: datetime) -> Tuple[bool, Optional[DeparturePattern]]:
        """Check if a departure at this time matches a known pattern."""
        day = timestamp.weekday()
        current_time = timestamp.time()

        for pattern in self.patterns:
            if pattern.day_of_week == day:
                if pattern.time_window_start <= current_time <= pattern.time_window_end:
                    return True, pattern

        return False, None
```

### Pattern Storage

```yaml
# data/temporal_patterns.yaml
patterns:
  - day_of_week: 4  # Friday
    time_window_start: "16:30"
    time_window_end: "17:30"
    occurrence_count: 8
    last_occurred: "2025-01-17T16:45:00"
    confidence: 0.85
    typical_duration_minutes: 45

  - day_of_week: 0  # Monday
    time_window_start: "08:00"
    time_window_end: "08:30"
    occurrence_count: 12
    last_occurred: "2025-01-20T08:15:00"
    confidence: 0.95
    typical_duration_minutes: 540  # 9 hours (work day)

departures:
  - timestamp: "2025-01-17T16:45:00"
    was_owner: true
    return_time: "2025-01-17T17:30:00"
    duration_minutes: 45

  - timestamp: "2025-01-20T08:15:00"
    was_owner: true
    return_time: "2025-01-20T17:20:00"
    duration_minutes: 545
```

---

## 6. Alert Decision Logic

### When to Alert

```python
def should_alert_departure(
    departure_signals: List[DepartureSignal],
    actor_data: Optional[dict],
    owner_profile: OwnerProfile,
    pattern_db: TemporalPatternDB,
    current_time: datetime
) -> Tuple[bool, str, float]:
    """
    Decide whether to send a departure alert.

    Returns: (should_alert, reason, urgency)
    """
    if not departure_signals:
        return False, "", 0.0

    # Calculate overall departure confidence
    max_confidence = max(s.confidence for s in departure_signals)

    # Check if this is an expected departure time
    is_expected, pattern = pattern_db.is_expected_departure(current_time)

    # Check if actor matches owner
    is_owner = False
    owner_confidence = 0.0
    if actor_data:
        is_owner, owner_confidence = is_likely_owner(actor_data, owner_profile)

    # Decision matrix
    if is_owner and owner_confidence > 0.85:
        # High confidence owner - still alert but low urgency (for learning)
        return True, "owner_departure_confirmed", 0.2

    elif is_expected and pattern.confidence > 0.8:
        # Expected time window - alert with medium urgency
        return True, "expected_departure_time", 0.4

    elif is_owner and owner_confidence > 0.6:
        # Probable owner - alert with medium urgency
        return True, "probable_owner_departure", 0.5

    elif actor_data and not is_owner:
        # Someone else is taking the car!
        return True, "unknown_person_departure", 1.0

    else:
        # Departure detected, no actor identified
        return True, "unidentified_departure", 0.8

    return False, "", 0.0
```

### Alert Message Format

```python
def format_departure_alert(
    reason: str,
    urgency: float,
    signals: List[DepartureSignal],
    actor_data: Optional[dict],
    pattern: Optional[DeparturePattern]
) -> str:
    """Format alert message for Telegram."""

    urgency_emoji = {
        (0.0, 0.3): "ℹ️",      # Info
        (0.3, 0.6): "⚠️",      # Warning
        (0.6, 0.9): "🚨",      # Alert
        (0.9, 1.1): "🚨🚨🚨",  # Critical
    }

    emoji = next(e for (lo, hi), e in urgency_emoji.items() if lo <= urgency < hi)

    messages = {
        "owner_departure_confirmed": f"{emoji} Car departing - owner identified",
        "expected_departure_time": f"{emoji} Car departing - expected time window",
        "probable_owner_departure": f"{emoji} Car departing - probably you",
        "unknown_person_departure": f"{emoji} CAR DEPARTING - UNKNOWN PERSON",
        "unidentified_departure": f"{emoji} Car departing - could not identify driver",
    }

    msg = messages.get(reason, f"{emoji} Car activity detected")

    # Add signal details
    signal_types = [s.signal_type for s in signals]
    msg += f"\n\nSignals: {', '.join(signal_types)}"

    if pattern:
        msg += f"\n\nMatches pattern: {pattern.day_of_week} {pattern.time_window_start}-{pattern.time_window_end}"

    msg += "\n\nReply 'me' if this was you."

    return msg
```

---

## 7. Feedback Processing

### Enhanced Telegram Response Handler

```python
class FeedbackHandler:
    """Process owner feedback to improve detection."""

    OWNER_CONFIRMATIONS = ['me', 'mine', 'owner', 'yes', 'that was me', "it's me", "its me"]
    NOT_OWNER = ['no', 'not me', 'theft', 'stolen', 'alert', 'unknown']

    def __init__(self, owner_profile: OwnerProfile, pattern_db: TemporalPatternDB):
        self.owner_profile = owner_profile
        self.pattern_db = pattern_db
        self.pending_events = {}  # event_id -> event_data

    def register_departure_event(self, event_id: str, event_data: dict):
        """Register a departure event awaiting feedback."""
        self.pending_events[event_id] = {
            **event_data,
            'registered_at': time.time()
        }

    def process_feedback(self, event_id: str, response: str) -> dict:
        """Process user feedback on a departure event."""
        response_lower = response.lower().strip()
        event = self.pending_events.get(event_id)

        if not event:
            return {'status': 'event_not_found'}

        result = {'status': 'processed', 'actions': []}

        if any(conf in response_lower for conf in self.OWNER_CONFIRMATIONS):
            # Owner confirmed - update learning

            # 1. Add appearance sample if we captured the actor
            if event.get('actor_crop'):
                self.owner_profile.add_sample(event['actor_crop'], event['timestamp'])
                result['actions'].append('added_appearance_sample')

            # 2. Record departure pattern
            self.pattern_db.record_departure(
                timestamp=event['timestamp'],
                was_owner=True,
                duration=None  # Will be updated on return
            )
            result['actions'].append('recorded_departure_pattern')

            # 3. Mark event as owner-confirmed
            event['confirmed_owner'] = True
            result['actions'].append('marked_as_owner')

        elif any(neg in response_lower for neg in self.NOT_OWNER):
            # Not owner - this is concerning!
            event['confirmed_owner'] = False
            result['actions'].append('marked_as_not_owner')
            result['alert'] = 'unauthorized_departure'

        return result

    def process_return(self, return_time: datetime):
        """Process car return - update duration patterns."""
        # Find most recent unresolved departure
        for event_id, event in sorted(
            self.pending_events.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        ):
            if event.get('confirmed_owner') and not event.get('return_processed'):
                departure_time = event['timestamp']
                duration = (return_time - departure_time).total_seconds() / 60

                # Update pattern with duration
                self.pattern_db.update_last_departure_duration(int(duration))
                event['return_processed'] = True
                event['duration_minutes'] = duration
                break
```

---

## 8. Integration with Existing Pipeline

### Modified Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frame Input                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0: Presence Check (NEW)                                   │
│  - Check current presence state                                  │
│  - Detect light conditions                                       │
│  - Route to appropriate detection path                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
        ┌─────────┐    ┌───────────┐   ┌──────────┐
        │ PRESENT │    │  ABSENT   │   │DEPARTING/│
        │  Path   │    │   Path    │   │RETURNING │
        └────┬────┘    └─────┬─────┘   └────┬─────┘
             │               │              │
             ▼               ▼              ▼
┌────────────────┐  ┌─────────────┐  ┌──────────────────┐
│ Stage 1: Car   │  │ Wait for    │  │ Departure/Return │
│ Detection      │  │ car return  │  │ Analysis         │
│ (existing)     │  │ signals     │  │ (NEW)            │
└───────┬────────┘  └─────────────┘  └────────┬─────────┘
        │                                     │
        ▼                                     ▼
┌────────────────┐                   ┌──────────────────┐
│ Stage 2:       │                   │ Owner ID &       │
│ Proximity      │                   │ Pattern Check    │
│ (existing)     │                   │ (NEW)            │
└───────┬────────┘                   └────────┬─────────┘
        │                                     │
        ▼                                     ▼
┌────────────────┐                   ┌──────────────────┐
│ Stage 3:       │                   │ Alert Decision   │
│ Contact        │                   │ (NEW)            │
│ (existing)     │                   └────────┬─────────┘
└───────┬────────┘                            │
        │                                     │
        └──────────────┬──────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Stage 4:       │
              │ Alert/Record   │
              │ (enhanced)     │
              └────────────────┘
```

### New Configuration Options

```yaml
# config/config.yaml additions

presence_tracking:
  enabled: true

  # Baseline establishment
  baseline_stability_frames: 10  # Frames before baseline is set
  position_tolerance_px: 50      # Allowed drift from baseline
  area_tolerance_pct: 0.15       # Allowed size change (15%)

  # Departure detection
  departure_confirmation_frames: 5   # Frames of signals before DEPARTING state
  absence_confirmation_frames: 30    # Frames without car before ABSENT state

  # Low-light detection
  enable_light_detection: true
  tail_light_min_area: 50           # Minimum blob size for light detection
  motion_threshold: 0.02            # 2% of zone must change
  significant_motion_threshold: 0.10 # 10% = major movement

temporal_patterns:
  enabled: true
  min_occurrences_for_pattern: 2    # Need 2+ events to form pattern
  time_cluster_window_minutes: 30   # Events within 30min = same pattern
  pattern_confidence_base: 0.5
  pattern_confidence_per_occurrence: 0.1
  max_pattern_confidence: 0.95

departure_alerts:
  # Alert urgency thresholds
  owner_confirmed_urgency: 0.2
  expected_time_urgency: 0.4
  probable_owner_urgency: 0.5
  unknown_person_urgency: 1.0
  unidentified_urgency: 0.8

  # Always alert even for confirmed owner (for learning)
  alert_on_owner_departure: true

  # Suppress alerts during expected patterns after N confirmations
  suppress_after_confirmations: 10
```

---

## 9. Implementation Phases

### Phase 1: Presence State Machine (Foundation)
- [ ] Implement `PresenceState` enum and `PresenceTracker` class
- [ ] Add baseline establishment logic
- [ ] Integrate state machine into pipeline
- [ ] Add state persistence (survive restarts)

### Phase 2: Low-Light Detection
- [ ] Implement light detection in car zone
- [ ] Implement motion detection in car zone
- [ ] Add light condition detection (daylight/twilight/night)
- [ ] Combine signals for departure detection

### Phase 3: Enhanced Owner Identification
- [ ] Capture actor at departure
- [ ] Integrate with existing owner profile matching
- [ ] Improve feature extraction for better matching

### Phase 4: Temporal Pattern Learning
- [ ] Implement `TemporalPatternDB`
- [ ] Add pattern clustering algorithm
- [ ] Store and load patterns from YAML
- [ ] Integrate pattern checking into alert decisions

### Phase 5: Feedback Loop Enhancement
- [ ] Enhance Telegram response handling
- [ ] Add return detection and duration tracking
- [ ] Implement pattern updates from feedback
- [ ] Add suppression logic for learned patterns

### Phase 6: Testing & Tuning
- [ ] Test daylight departure detection
- [ ] Test low-light departure detection
- [ ] Tune thresholds for false positive/negative balance
- [ ] Long-term pattern learning validation

---

## 10. Open Questions

1. **Multi-car households**: Should the system support tracking multiple vehicles?

2. **Guest drivers**: How to handle known guests who might use the car legitimately?

3. **Pattern decay**: Should old patterns lose confidence over time if not reinforced?

4. **Privacy**: Should departure patterns be encrypted at rest?

5. **Remote arming**: Should there be a way to "arm" the system when you know you won't be using the car?

---

## Summary

This design extends pi-car-monitor from a "contact detection" system to a comprehensive "car presence and activity" system. Key additions:

1. **State machine** tracking car presence (UNKNOWN → PRESENT → DEPARTING → ABSENT → RETURNING)
2. **Dynamic baseline** that updates when car returns
3. **Low-light detection** using tail lights, motion, and zone analysis
4. **Temporal pattern learning** that reduces alerts for expected usage
5. **Enhanced feedback loop** that learns from owner confirmations

The goal is to answer not just "Is someone messing with my car?" but also "Has my car moved, and should I be concerned?"
