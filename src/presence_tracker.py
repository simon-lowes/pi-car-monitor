"""
Presence Tracker - Car Presence State Machine
==============================================
Tracks whether the owner's car is present, departing, absent, or returning.

This module manages:
1. Car presence state (UNKNOWN -> PRESENT -> DEPARTING -> ABSENT -> RETURNING)
2. Dynamic position baseline establishment and updates
3. Departure signal detection
4. State persistence across restarts

Integrates with the existing detection pipeline as "Stage 0".
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import yaml

import cv2
import numpy as np

from low_light_detector import LowLightDetector, LightCondition
from person_features import PersonFeatureExtractor, DepartureActorTracker, DepartureActor

logger = logging.getLogger(__name__)


class PresenceState(Enum):
    """Car presence states."""
    UNKNOWN = auto()      # Startup, no baseline established
    PRESENT = auto()      # Car in known position, baseline established
    DEPARTING = auto()    # Motion/lights detected, car may be leaving
    ABSENT = auto()       # Car confirmed not in frame
    RETURNING = auto()    # Car detected entering zone after absence


@dataclass
class CarBaseline:
    """Established car position and appearance baseline."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    center: Tuple[int, int]
    area: int
    established_at: float  # Unix timestamp
    light_conditions: str  # 'daylight', 'low_light', 'night'
    confidence: float
    frame_width: int
    frame_height: int

    # Tolerance for position drift
    position_tolerance: int = 50  # pixels
    area_tolerance_pct: float = 0.15  # 15% size change allowed

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            'bbox': {
                'x1': self.bbox[0],
                'y1': self.bbox[1],
                'x2': self.bbox[2],
                'y2': self.bbox[3]
            },
            'center': list(self.center),
            'area': self.area,
            'established_at': datetime.fromtimestamp(self.established_at).isoformat(),
            'established_at_unix': self.established_at,
            'light_conditions': self.light_conditions,
            'confidence': self.confidence,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'position_tolerance': self.position_tolerance,
            'area_tolerance_pct': self.area_tolerance_pct
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CarBaseline':
        """Create from dictionary (YAML deserialization)."""
        bbox_data = data['bbox']
        return cls(
            bbox=(bbox_data['x1'], bbox_data['y1'], bbox_data['x2'], bbox_data['y2']),
            center=tuple(data['center']),
            area=data['area'],
            established_at=data.get('established_at_unix', time.time()),
            light_conditions=data['light_conditions'],
            confidence=data['confidence'],
            frame_width=data['frame_width'],
            frame_height=data['frame_height'],
            position_tolerance=data.get('position_tolerance', 50),
            area_tolerance_pct=data.get('area_tolerance_pct', 0.15)
        )


@dataclass
class DepartureSignal:
    """A signal indicating potential car departure."""
    timestamp: float
    signal_type: str  # 'lights', 'motion', 'position_shift', 'car_missing'
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PresenceEvent:
    """Record of a presence state change."""
    timestamp: float
    from_state: PresenceState
    to_state: PresenceState
    trigger: str
    details: Dict[str, Any] = field(default_factory=dict)


class PresenceTracker:
    """
    Tracks car presence state and manages position baseline.

    This is the core state machine for departure/return detection.
    """

    def __init__(self, config: dict, baseline_path: str = "data/car_baseline.yaml"):
        self.config = config.get('presence_tracking', {})
        self.baseline_path = Path(baseline_path)

        # Configuration
        self.baseline_stability_frames = self.config.get('baseline_stability_frames', 10)
        self.position_tolerance_px = self.config.get('position_tolerance_px', 50)
        self.area_tolerance_pct = self.config.get('area_tolerance_pct', 0.15)
        self.departure_confirmation_frames = self.config.get('departure_confirmation_frames', 5)
        self.absence_confirmation_frames = self.config.get('absence_confirmation_frames', 30)
        self.return_stability_frames = self.config.get('return_stability_frames', 10)

        # State
        self.state = PresenceState.UNKNOWN
        self.baseline: Optional[CarBaseline] = None
        self.previous_baseline: Optional[CarBaseline] = None

        # Counters for state stability
        self.frames_in_state = 0
        self.frames_car_detected = 0
        self.frames_car_missing = 0
        self.frames_departure_signals = 0

        # Departure tracking
        self.departure_started_at: Optional[float] = None
        self.departure_signals: List[DepartureSignal] = []

        # Event history
        self.events: List[PresenceEvent] = []
        self.max_events = 100

        # Frame storage for baseline snapshot
        self._baseline_frame: Optional[np.ndarray] = None

        # Previous frame for motion detection
        self._prev_frame: Optional[np.ndarray] = None

        # Initialize low-light detector for enhanced detection
        self.low_light_detector = LowLightDetector(config)

        # Initialize departure actor tracker for owner identification
        self.actor_tracker = DepartureActorTracker(config)

        # Store departure actor for feedback learning
        self.last_departure_actor: Optional[DepartureActor] = None

        # Load persisted baseline
        self._load_baseline()

        logger.info(f"PresenceTracker initialized, state={self.state.name}, "
                   f"baseline={'loaded' if self.baseline else 'none'}")

    def _load_baseline(self):
        """Load persisted baseline from file."""
        if not self.baseline_path.exists():
            logger.info("No baseline file found, starting fresh")
            return

        try:
            with open(self.baseline_path, 'r') as f:
                data = yaml.safe_load(f)

            if data and 'baseline' in data:
                self.baseline = CarBaseline.from_dict(data['baseline'])
                self.baseline.position_tolerance = self.position_tolerance_px
                self.baseline.area_tolerance_pct = self.area_tolerance_pct

                # If we have a baseline, start in UNKNOWN but ready to verify
                logger.info(f"Loaded baseline from {self.baseline_path}: "
                           f"center={self.baseline.center}, area={self.baseline.area}")

                # Load history if present
                if 'history' in data:
                    # Keep for reference but don't load into events
                    pass

        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")

    def _save_baseline(self):
        """Persist baseline to file."""
        try:
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'baseline': self.baseline.to_dict() if self.baseline else None,
                'history': [],
                'last_updated': datetime.now().isoformat()
            }

            # Add previous baseline to history if different
            if self.previous_baseline:
                data['history'].append(self.previous_baseline.to_dict())

            with open(self.baseline_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)

            logger.debug(f"Baseline saved to {self.baseline_path}")

        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")

    def _save_baseline_snapshot(self, frame: np.ndarray):
        """Save a snapshot of the frame when baseline was established."""
        try:
            snapshot_dir = self.baseline_path.parent / "baselines"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = snapshot_dir / f"baseline_{timestamp}.jpg"

            cv2.imwrite(str(snapshot_path), frame)
            logger.info(f"Baseline snapshot saved to {snapshot_path}")

        except Exception as e:
            logger.warning(f"Failed to save baseline snapshot: {e}")

    def _transition_state(self, new_state: PresenceState, trigger: str, details: dict = None):
        """Transition to a new state and record the event."""
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state
        self.frames_in_state = 0

        event = PresenceEvent(
            timestamp=time.time(),
            from_state=old_state,
            to_state=new_state,
            trigger=trigger,
            details=details or {}
        )
        self.events.append(event)

        # Trim event history
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        logger.info(f"State transition: {old_state.name} -> {new_state.name} (trigger: {trigger})")

    def _detect_light_conditions(self, frame: np.ndarray) -> str:
        """Estimate current lighting conditions from frame brightness."""
        condition = self.low_light_detector.detect_light_condition(frame)
        return condition.name.lower()

    def _calculate_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _calculate_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """Calculate area of bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _euclidean_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def is_car_in_baseline_position(
        self,
        current_bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, float]:
        """
        Check if car is in its expected baseline position.

        Returns:
            (is_in_position, confidence)
        """
        if not self.baseline:
            return False, 0.0

        current_center = self._calculate_center(current_bbox)
        current_area = self._calculate_area(current_bbox)

        # Distance from baseline center
        distance = self._euclidean_distance(current_center, self.baseline.center)

        # Area ratio
        if self.baseline.area == 0:
            return False, 0.0

        area_ratio = current_area / self.baseline.area

        # Position score (1.0 = perfect match, 0.0 = outside tolerance)
        position_score = max(0, 1 - (distance / self.baseline.position_tolerance))

        # Area score
        area_deviation = abs(1 - area_ratio)
        area_score = max(0, 1 - (area_deviation / self.baseline.area_tolerance_pct))

        # Combined confidence
        confidence = (position_score * 0.6) + (area_score * 0.4)

        in_position = (
            distance <= self.baseline.position_tolerance and
            (1 - self.baseline.area_tolerance_pct) <= area_ratio <= (1 + self.baseline.area_tolerance_pct)
        )

        return in_position, confidence

    def establish_baseline(
        self,
        frame: np.ndarray,
        car_bbox: Tuple[int, int, int, int],
        confidence: float
    ) -> bool:
        """
        Establish or update the car position baseline.

        Should be called when car is stably detected in daylight.

        Returns:
            True if baseline was established/updated
        """
        light_conditions = self._detect_light_conditions(frame)

        # Prefer establishing baseline in good lighting
        if light_conditions == 'night' and not self.baseline:
            logger.debug("Skipping baseline establishment in night conditions")
            return False

        frame_h, frame_w = frame.shape[:2]

        # Store previous baseline before updating
        if self.baseline:
            self.previous_baseline = self.baseline

        self.baseline = CarBaseline(
            bbox=car_bbox,
            center=self._calculate_center(car_bbox),
            area=self._calculate_area(car_bbox),
            established_at=time.time(),
            light_conditions=light_conditions,
            confidence=confidence,
            frame_width=frame_w,
            frame_height=frame_h,
            position_tolerance=self.position_tolerance_px,
            area_tolerance_pct=self.area_tolerance_pct
        )

        # Save snapshot and persist
        self._save_baseline_snapshot(frame)
        self._save_baseline()

        logger.info(f"Baseline established: center={self.baseline.center}, "
                   f"area={self.baseline.area}, light={light_conditions}")

        return True

    def detect_zone_motion(
        self,
        frame: np.ndarray,
        threshold: int = 25
    ) -> Dict[str, Any]:
        """
        Detect motion within the car zone using frame differencing.

        Returns dict with motion metrics.
        """
        if self._prev_frame is None or not self.baseline:
            self._prev_frame = frame.copy()
            return {'motion_detected': False, 'motion_ratio': 0.0}

        x1, y1, x2, y2 = self.baseline.bbox

        # Ensure bounds are valid
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return {'motion_detected': False, 'motion_ratio': 0.0}

        prev_zone = cv2.cvtColor(self._prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        curr_zone = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        prev_zone = cv2.GaussianBlur(prev_zone, (21, 21), 0)
        curr_zone = cv2.GaussianBlur(curr_zone, (21, 21), 0)

        # Frame difference
        diff = cv2.absdiff(prev_zone, curr_zone)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Calculate motion metrics
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]

        if total_pixels == 0:
            motion_ratio = 0.0
        else:
            motion_ratio = motion_pixels / total_pixels

        # Find motion centroid
        M = cv2.moments(thresh)
        motion_center = None
        if M["m00"] > 0:
            motion_center = (
                int(M["m10"] / M["m00"]) + x1,
                int(M["m01"] / M["m00"]) + y1
            )

        # Update previous frame
        self._prev_frame = frame.copy()

        return {
            'motion_detected': motion_ratio > 0.02,  # 2% of zone changed
            'motion_ratio': motion_ratio,
            'motion_center': motion_center,
            'is_significant': motion_ratio > 0.10  # 10% = major movement
        }

    def detect_lights_in_zone(
        self,
        frame: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect illumination (tail lights, headlights) within the car zone.

        Works in low-light conditions where object detection may fail.
        """
        if not self.baseline:
            return []

        x1, y1, x2, y2 = self.baseline.bbox

        # Ensure bounds are valid
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return []

        zone = frame[y1:y2, x1:x2]

        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)

        events = []
        min_area = self.config.get('tail_light_min_area', 50)

        # Tail lights: Red, high saturation, high value
        red_mask_low = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask_high = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_mask = red_mask_low | red_mask_high

        # Reverse/headlights: White/bright, low saturation, high value
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))

        for mask, light_type, color in [
            (red_mask, 'tail_light', 'red'),
            (white_mask, 'reverse_light', 'white')
        ]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"]) + x1
                        cy = int(M["m01"] / M["m00"]) + y1
                        zone_area = zone.shape[0] * zone.shape[1]
                        intensity = min(1.0, (area / zone_area) * 10) if zone_area > 0 else 0

                        events.append({
                            'type': light_type,
                            'location': (cx, cy),
                            'intensity': intensity,
                            'color': color,
                            'area': area
                        })

        return events

    def process_frame(
        self,
        frame: np.ndarray,
        car_detected: bool,
        car_bbox: Optional[Tuple[int, int, int, int]],
        car_confidence: float,
        car_stable: bool,
        person_detections: Optional[List[Dict]] = None,
        pose_detections: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process a frame and update presence state.

        This is the main entry point called by the pipeline.

        Args:
            frame: Current video frame
            car_detected: Whether the car was detected by CarDetector
            car_bbox: Bounding box if detected
            car_confidence: Detection confidence
            car_stable: Whether detection is stable (3+ consecutive frames)
            person_detections: Optional list of person detections with 'bbox'
            pose_detections: Optional list of pose detections with 'bbox' and 'keypoints'

        Returns:
            Dict with:
                - state: Current PresenceState
                - state_changed: Whether state just changed
                - baseline: Current baseline if any
                - departure_signals: Any departure signals detected
                - should_alert: Whether an alert should be sent
                - alert_reason: Reason for alert if any
                - departure_actor: DepartureActor if detected during departure
        """
        self.frames_in_state += 1
        timestamp = time.time()

        result = {
            'state': self.state,
            'state_changed': False,
            'baseline': self.baseline,
            'departure_signals': [],
            'should_alert': False,
            'alert_reason': None,
            'light_conditions': self._detect_light_conditions(frame),
            'departure_actor': None,
            'owner_match_confidence': None
        }

        # Track actors during PRESENT and DEPARTING states if we have person detections
        if (self.state in [PresenceState.PRESENT, PresenceState.DEPARTING] and
            person_detections and car_bbox):
            actors = self.actor_tracker.process_frame(
                frame=frame,
                person_detections=person_detections,
                pose_detections=pose_detections,
                car_bbox=car_bbox,
                timestamp=timestamp
            )

            # Get primary actor (most likely to be getting in car)
            primary_actor = self.actor_tracker.get_primary_actor()
            if primary_actor:
                result['departure_actor'] = primary_actor
                self.last_departure_actor = primary_actor

        # State machine logic
        if self.state == PresenceState.UNKNOWN:
            result = self._handle_unknown_state(frame, car_detected, car_bbox, car_confidence, car_stable, result)

        elif self.state == PresenceState.PRESENT:
            result = self._handle_present_state(frame, car_detected, car_bbox, car_confidence, car_stable, result)

        elif self.state == PresenceState.DEPARTING:
            result = self._handle_departing_state(frame, car_detected, car_bbox, car_confidence, car_stable, result)

        elif self.state == PresenceState.ABSENT:
            result = self._handle_absent_state(frame, car_detected, car_bbox, car_confidence, car_stable, result)

        elif self.state == PresenceState.RETURNING:
            result = self._handle_returning_state(frame, car_detected, car_bbox, car_confidence, car_stable, result)

        return result

    def _handle_unknown_state(
        self,
        frame: np.ndarray,
        car_detected: bool,
        car_bbox: Optional[Tuple[int, int, int, int]],
        car_confidence: float,
        car_stable: bool,
        result: dict
    ) -> dict:
        """Handle UNKNOWN state - waiting for initial baseline."""

        if car_detected and car_stable:
            self.frames_car_detected += 1

            # Need stable detection for N frames to establish baseline
            if self.frames_car_detected >= self.baseline_stability_frames:
                light_conditions = self._detect_light_conditions(frame)

                # Establish baseline
                if self.establish_baseline(frame, car_bbox, car_confidence):
                    self._transition_state(
                        PresenceState.PRESENT,
                        'baseline_established',
                        {'light_conditions': light_conditions}
                    )
                    result['state'] = self.state
                    result['state_changed'] = True
                    result['baseline'] = self.baseline

                self.frames_car_detected = 0
        else:
            # Reset counter if detection lost
            self.frames_car_detected = max(0, self.frames_car_detected - 1)

            # If we have a saved baseline, check if car is there
            if self.baseline and car_detected and car_bbox:
                in_position, confidence = self.is_car_in_baseline_position(car_bbox)
                if in_position and confidence > 0.7:
                    self._transition_state(
                        PresenceState.PRESENT,
                        'baseline_verified',
                        {'position_confidence': confidence}
                    )
                    result['state'] = self.state
                    result['state_changed'] = True

        return result

    def _handle_present_state(
        self,
        frame: np.ndarray,
        car_detected: bool,
        car_bbox: Optional[Tuple[int, int, int, int]],
        car_confidence: float,
        car_stable: bool,
        result: dict
    ) -> dict:
        """Handle PRESENT state - car is in known position."""

        departure_signals = []

        # Check for departure signals even if car is still detected
        if self.baseline:
            # Use enhanced low-light detector for comprehensive signal detection
            low_light_result = self.low_light_detector.detect_departure_signals(
                frame=frame,
                car_zone=self.baseline.bbox,
                car_detected_by_yolo=car_detected
            )

            # Process detected lights
            if low_light_result['light_pattern']:
                pattern = low_light_result['light_pattern']

                if pattern.pattern_type == 'reversing':
                    departure_signals.append(DepartureSignal(
                        timestamp=time.time(),
                        signal_type='reverse_lights',
                        confidence=pattern.confidence,
                        details={
                            'pattern': pattern.pattern_type,
                            'light_count': len(pattern.lights),
                            'paired': len(low_light_result['light_pairs']) > 0
                        }
                    ))
                elif pattern.pattern_type in ['running', 'braking']:
                    # Check for paired tail lights (stronger signal)
                    tail_pairs = [p for p in low_light_result['light_pairs']]
                    if tail_pairs:
                        departure_signals.append(DepartureSignal(
                            timestamp=time.time(),
                            signal_type='tail_lights',
                            confidence=pattern.confidence,
                            details={
                                'pattern': pattern.pattern_type,
                                'paired': True,
                                'pair_count': len(tail_pairs)
                            }
                        ))
                elif pattern.pattern_type == 'door_open':
                    departure_signals.append(DepartureSignal(
                        timestamp=time.time(),
                        signal_type='interior_light',
                        confidence=pattern.confidence,
                        details={'pattern': pattern.pattern_type}
                    ))

            # Process motion analysis
            motion = low_light_result['motion']
            if motion.is_significant:
                motion_details = {
                    'motion_ratio': motion.motion_ratio,
                    'direction': motion.motion_direction,
                    'flow_magnitude': motion.flow_magnitude,
                    'is_car_sized': motion.is_car_sized
                }

                # Higher confidence for car-sized motion with direction
                motion_confidence = 0.6
                if motion.is_car_sized:
                    motion_confidence = 0.75
                if motion.motion_direction == 'toward_camera':
                    # Car backing out - strong signal
                    motion_confidence = 0.85

                departure_signals.append(DepartureSignal(
                    timestamp=time.time(),
                    signal_type='zone_motion',
                    confidence=motion_confidence,
                    details=motion_details
                ))

            # Store light condition in result for reference
            result['light_condition'] = low_light_result['light_condition'].name.lower()

        # Check if car position has shifted
        if car_detected and car_bbox and self.baseline:
            in_position, pos_confidence = self.is_car_in_baseline_position(car_bbox)
            if not in_position:
                departure_signals.append(DepartureSignal(
                    timestamp=time.time(),
                    signal_type='position_shift',
                    confidence=0.8,
                    details={
                        'expected_center': self.baseline.center,
                        'actual_center': self._calculate_center(car_bbox),
                        'position_confidence': pos_confidence
                    }
                ))

        # Check if car is no longer detected (in daylight)
        light_conditions = self._detect_light_conditions(frame)
        if not car_detected:
            if light_conditions == 'daylight':
                departure_signals.append(DepartureSignal(
                    timestamp=time.time(),
                    signal_type='car_missing',
                    confidence=0.9,
                    details={'light_conditions': light_conditions}
                ))
            self.frames_car_missing += 1
        else:
            self.frames_car_missing = 0

        # Process departure signals
        if departure_signals:
            self.departure_signals.extend(departure_signals)
            self.frames_departure_signals += 1
            result['departure_signals'] = departure_signals

            # Require confirmation frames before transitioning
            if self.frames_departure_signals >= self.departure_confirmation_frames:
                self.departure_started_at = time.time()
                self._transition_state(
                    PresenceState.DEPARTING,
                    'departure_signals_detected',
                    {
                        'signals': [s.signal_type for s in self.departure_signals[-5:]],
                        'light_conditions': light_conditions
                    }
                )
                result['state'] = self.state
                result['state_changed'] = True
                result['should_alert'] = True
                result['alert_reason'] = 'departure_detected'
        else:
            self.frames_departure_signals = 0

        return result

    def _handle_departing_state(
        self,
        frame: np.ndarray,
        car_detected: bool,
        car_bbox: Optional[Tuple[int, int, int, int]],
        car_confidence: float,
        car_stable: bool,
        result: dict
    ) -> dict:
        """Handle DEPARTING state - car is in the process of leaving."""

        light_conditions = self._detect_light_conditions(frame)

        # Check if car is back in position (false alarm)
        if car_detected and car_bbox and self.baseline:
            in_position, pos_confidence = self.is_car_in_baseline_position(car_bbox)
            if in_position and pos_confidence > 0.8:
                self.frames_car_detected += 1
                if self.frames_car_detected >= 5:
                    # False alarm - car didn't actually leave
                    self._transition_state(
                        PresenceState.PRESENT,
                        'departure_cancelled',
                        {'position_confidence': pos_confidence}
                    )
                    result['state'] = self.state
                    result['state_changed'] = True
                    self.frames_car_detected = 0
                    self.departure_signals = []
                    self.departure_started_at = None
                    return result

        self.frames_car_detected = 0

        # Check if car is now absent
        if not car_detected:
            self.frames_car_missing += 1

            if self.frames_car_missing >= self.absence_confirmation_frames:
                self._transition_state(
                    PresenceState.ABSENT,
                    'car_left',
                    {
                        'departure_started': self.departure_started_at,
                        'light_conditions': light_conditions
                    }
                )
                result['state'] = self.state
                result['state_changed'] = True
                self.frames_car_missing = 0
        else:
            # Still detecting something, but might be partially visible
            self.frames_car_missing = max(0, self.frames_car_missing - 1)

        return result

    def _handle_absent_state(
        self,
        frame: np.ndarray,
        car_detected: bool,
        car_bbox: Optional[Tuple[int, int, int, int]],
        car_confidence: float,
        car_stable: bool,
        result: dict
    ) -> dict:
        """Handle ABSENT state - car is not present."""

        if car_detected:
            self.frames_car_detected += 1

            # Car is coming back
            if self.frames_car_detected >= 3:
                self._transition_state(
                    PresenceState.RETURNING,
                    'car_detected',
                    {'confidence': car_confidence}
                )
                result['state'] = self.state
                result['state_changed'] = True
                result['should_alert'] = True
                result['alert_reason'] = 'car_returning'
                self.frames_car_detected = 0
        else:
            self.frames_car_detected = 0

        return result

    def _handle_returning_state(
        self,
        frame: np.ndarray,
        car_detected: bool,
        car_bbox: Optional[Tuple[int, int, int, int]],
        car_confidence: float,
        car_stable: bool,
        result: dict
    ) -> dict:
        """Handle RETURNING state - car is coming back."""

        if car_detected and car_stable and car_bbox:
            self.frames_car_detected += 1

            # Car is stably back - update baseline and transition to PRESENT
            if self.frames_car_detected >= self.return_stability_frames:
                light_conditions = self._detect_light_conditions(frame)

                # Update baseline with new position
                if light_conditions != 'night':
                    self.establish_baseline(frame, car_bbox, car_confidence)

                self._transition_state(
                    PresenceState.PRESENT,
                    'car_returned',
                    {
                        'light_conditions': light_conditions,
                        'baseline_updated': light_conditions != 'night'
                    }
                )
                result['state'] = self.state
                result['state_changed'] = True
                result['should_alert'] = True
                result['alert_reason'] = 'car_returned'
                result['baseline'] = self.baseline
                self.frames_car_detected = 0

                # Clear departure tracking
                self.departure_signals = []
                self.departure_started_at = None
        else:
            # Detection lost while returning
            self.frames_car_detected = max(0, self.frames_car_detected - 1)
            if self.frames_car_detected == 0:
                # Lost car again, go back to absent
                self._transition_state(
                    PresenceState.ABSENT,
                    'car_lost_during_return',
                    {}
                )
                result['state'] = self.state
                result['state_changed'] = True

        return result

    def get_status(self) -> dict:
        """Get current tracker status for debugging/monitoring."""
        return {
            'state': self.state.name,
            'frames_in_state': self.frames_in_state,
            'has_baseline': self.baseline is not None,
            'baseline_age_seconds': (
                time.time() - self.baseline.established_at
                if self.baseline else None
            ),
            'departure_signals_count': len(self.departure_signals),
            'recent_events': [
                {
                    'from': e.from_state.name,
                    'to': e.to_state.name,
                    'trigger': e.trigger,
                    'timestamp': e.timestamp
                }
                for e in self.events[-5:]
            ]
        }
