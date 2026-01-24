"""
Contact Classifier - Stage 3
=============================
Determines if a person/object is making physical contact with the car.

This is the critical stage that triggers recording. It uses:
1. Pose estimation to track body parts (especially hands)
2. Proximity analysis to detect contact distance
3. Motion vectors to detect touch/impact events

The goal is to distinguish between:
- Someone walking past (no contact) → DON'T record
- Someone walking their dog past (no contact) → DON'T record
- Someone's hand touching the car → RECORD
- Another vehicle bumping the car → RECORD
- Someone leaning on the car → RECORD
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class ContactType(Enum):
    """Types of contact that can be detected."""
    NONE = "none"
    PROXIMITY = "proximity"      # Close but not touching
    HAND_TOUCH = "hand_touch"    # Hand making contact
    BODY_LEAN = "body_lean"      # Person leaning on car
    VEHICLE_CONTACT = "vehicle"  # Another vehicle touching
    IMPACT = "impact"            # Sudden contact (possible collision)
    UNKNOWN = "unknown"          # Contact detected but type unclear


@dataclass
class TrackedPerson:
    """A person being tracked near the car."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    keypoints: Optional[dict]  # Pose estimation keypoints
    distance_to_car: float
    frames_in_proximity: int
    hand_positions: List[Tuple[int, int]]  # Last N hand positions


@dataclass
class TrackedVehicle:
    """A vehicle being tracked near the car."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    bbox_history: List[Tuple[int, int, int, int]]  # Last N bboxes for motion detection
    frames_tracked: int
    frames_overlapping: int  # Consecutive frames with overlap
    last_alert_timestamp: float  # For cooldown
    is_stationary: bool  # True if vehicle hasn't moved significantly


@dataclass
class ContactEvent:
    """A detected contact event."""
    contact_type: ContactType
    confidence: float
    location: Tuple[int, int]  # Where on the car
    actor_id: Optional[int]    # Track ID of person/vehicle involved
    timestamp: float
    duration: float = 0.0      # How long contact lasted


class ContactClassifier:
    """
    Classifies whether detected proximity constitutes physical contact.
    
    This is the most important component for privacy - it ensures we
    only record when there's genuine contact with the car, not just
    people passing by.
    """
    
    def __init__(self, config: dict, car_bbox_getter):
        """
        Args:
            config: Main configuration dict
            car_bbox_getter: Callable that returns current car bounding box
        """
        self.config = config
        self.get_car_bbox = car_bbox_getter
        
        detection_config = config.get("detection", {})
        self.contact_confidence = detection_config.get("contact_confidence", 0.7)
        self.dwell_time = detection_config.get("contact_dwell_time", 1.5)
        self.proximity_buffer = config.get("zones", {}).get("proximity_buffer", 50)

        # Vehicle contact settings (to reduce false positives from parked cars)
        vehicle_config = detection_config.get("vehicle_contact", {})
        self.vehicle_overlap_threshold = vehicle_config.get("overlap_threshold", 0.20)  # 20% overlap required
        self.vehicle_persistence_frames = vehicle_config.get("persistence_frames", 5)  # Must overlap for 5 frames
        self.vehicle_cooldown_seconds = vehicle_config.get("cooldown_seconds", 60.0)  # 60s between alerts for same vehicle
        self.vehicle_stationary_threshold = vehicle_config.get("stationary_threshold", 15)  # Pixels of movement to be "moving"
        self.vehicle_stationary_frames = vehicle_config.get("stationary_frames", 30)  # Frames to determine if stationary

        # Tracking state
        self.tracked_persons: dict[int, TrackedPerson] = {}
        self.tracked_vehicles: dict[int, TrackedVehicle] = {}  # Track other vehicles
        self.active_contacts: List[ContactEvent] = []
        self.contact_history = deque(maxlen=100)

        # Motion analysis
        self.prev_frame = None
        self.motion_threshold = 25  # Pixel difference threshold for motion
        
        # Keypoint indices for pose estimation (COCO format)
        self.KEYPOINTS = {
            'left_wrist': 9,
            'right_wrist': 10,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_shoulder': 5,
            'right_shoulder': 6,
        }
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: list,
        poses: Optional[list] = None,
        timestamp: float = 0.0
    ) -> List[ContactEvent]:
        """
        Process a frame to detect contact events.
        
        Args:
            frame: Current BGR frame
            detections: List of detected objects near car (from Stage 2)
            poses: Pose estimation results (optional, improves accuracy)
            timestamp: Frame timestamp
            
        Returns:
            List of ContactEvent if contact detected, empty list otherwise
        """
        car_bbox = self.get_car_bbox()
        if car_bbox is None:
            return []
        
        contact_events = []
        
        # Update tracked persons
        self._update_tracking(detections, poses)
        
        # Check each tracked person for contact
        for track_id, person in self.tracked_persons.items():
            contact = self._check_person_contact(frame, person, car_bbox, timestamp)
            if contact:
                contact_events.append(contact)
        
        # Check for vehicle-to-vehicle contact
        vehicle_contact = self._check_vehicle_contact(detections, car_bbox, timestamp)
        if vehicle_contact:
            contact_events.append(vehicle_contact)
        
        # Check for impact events (sudden motion near car)
        impact = self._check_impact_event(frame, car_bbox, timestamp)
        if impact:
            contact_events.append(impact)
        
        # Store for motion analysis
        self.prev_frame = frame.copy()
        
        # Update history
        for event in contact_events:
            self.contact_history.append(event)
        
        return contact_events
    
    def _update_tracking(self, detections: list, poses: Optional[list]):
        """Update tracked persons with new detections."""
        current_ids = set()

        for i, det in enumerate(detections):
            if det.get('class') != 'person':
                continue

            bbox = det['bbox']
            track_id = det.get('track_id', i)
            current_ids.add(track_id)

            # Match pose to this person by bbox overlap
            keypoints = None
            if poses:
                keypoints = self._match_pose_to_person(bbox, poses)

            # Calculate distance to car
            car_bbox = self.get_car_bbox()
            distance = self._calculate_distance(bbox, car_bbox) if car_bbox else float('inf')

            if track_id in self.tracked_persons:
                # Update existing
                person = self.tracked_persons[track_id]
                person.bbox = bbox
                person.keypoints = keypoints
                person.distance_to_car = distance
                person.frames_in_proximity += 1

                # Track hand positions for gesture analysis
                if keypoints:
                    hands = self._extract_hand_positions(keypoints)
                    person.hand_positions.extend(hands)
                    # Keep only last 30 positions
                    person.hand_positions = person.hand_positions[-30:]
            else:
                # New person
                hand_positions = []
                if keypoints:
                    hand_positions = self._extract_hand_positions(keypoints)

                self.tracked_persons[track_id] = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    keypoints=keypoints,
                    distance_to_car=distance,
                    frames_in_proximity=1,
                    hand_positions=hand_positions
                )

        # Remove persons who left the scene
        for track_id in list(self.tracked_persons.keys()):
            if track_id not in current_ids:
                del self.tracked_persons[track_id]

    def _match_pose_to_person(self, person_bbox: tuple, poses: list) -> Optional[dict]:
        """Match a pose detection to a person by bbox overlap."""
        best_match = None
        best_iou = 0.3  # Minimum IoU threshold

        for pose in poses:
            pose_bbox = pose.get('bbox')
            if not pose_bbox:
                continue

            # Calculate IoU between person bbox and pose bbox
            iou = self._calculate_iou(person_bbox, pose_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = pose.get('keypoints')

        return best_match

    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
    
    def _check_person_contact(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        car_bbox: tuple,
        timestamp: float
    ) -> Optional[ContactEvent]:
        """Check if a person is making contact with the car."""
        
        # Not close enough for contact
        if person.distance_to_car > self.proximity_buffer:
            return None
        
        # Check hand-to-car contact
        if person.keypoints:
            hand_contact = self._check_hand_contact(person, car_bbox)
            if hand_contact:
                location, confidence = hand_contact
                return ContactEvent(
                    contact_type=ContactType.HAND_TOUCH,
                    confidence=confidence,
                    location=location,
                    actor_id=person.track_id,
                    timestamp=timestamp
                )
        
        # Check body lean (prolonged very close proximity)
        if person.distance_to_car < self.proximity_buffer / 2:
            if person.frames_in_proximity > 30:  # ~2 seconds at 15fps
                # Check if person's body is overlapping with car bbox
                overlap = self._calculate_overlap(person.bbox, car_bbox)
                if overlap > 0.1:  # Significant overlap
                    return ContactEvent(
                        contact_type=ContactType.BODY_LEAN,
                        confidence=0.8,
                        location=self._get_overlap_center(person.bbox, car_bbox),
                        actor_id=person.track_id,
                        timestamp=timestamp
                    )
        
        return None
    
    def _check_hand_contact(
        self,
        person: TrackedPerson,
        car_bbox: tuple
    ) -> Optional[Tuple[Tuple[int, int], float]]:
        """Check if person's hand is touching the car."""
        if not person.keypoints:
            return None
        
        cx1, cy1, cx2, cy2 = car_bbox
        
        # Check both wrists
        for wrist_key in ['left_wrist', 'right_wrist']:
            idx = self.KEYPOINTS.get(wrist_key)
            if idx is None:
                continue
            
            keypoints = person.keypoints
            if isinstance(keypoints, dict):
                wrist = keypoints.get(wrist_key)
            elif isinstance(keypoints, (list, np.ndarray)) and len(keypoints) > idx:
                wrist = keypoints[idx]
            else:
                continue
            
            if wrist is None:
                continue
            
            wx, wy = wrist[:2] if len(wrist) >= 2 else (0, 0)
            confidence = wrist[2] if len(wrist) >= 3 else 0.5
            
            # Check if wrist is within or very close to car bbox
            margin = 10  # pixels
            if (cx1 - margin <= wx <= cx2 + margin and
                cy1 - margin <= wy <= cy2 + margin):
                
                # Additional check: is hand moving toward car?
                if len(person.hand_positions) >= 5:
                    moving_toward = self._check_hand_movement_toward_car(
                        person.hand_positions[-5:], car_bbox
                    )
                    if moving_toward:
                        confidence *= 1.2
                
                if confidence >= self.contact_confidence:
                    return ((int(wx), int(wy)), min(confidence, 1.0))
        
        return None
    
    def _check_hand_movement_toward_car(
        self,
        positions: List[Tuple[int, int]],
        car_bbox: tuple
    ) -> bool:
        """Check if hand is moving toward the car."""
        if len(positions) < 2:
            return False
        
        car_center = (
            (car_bbox[0] + car_bbox[2]) // 2,
            (car_bbox[1] + car_bbox[3]) // 2
        )
        
        # Calculate distances from first and last position to car
        def dist(pos):
            return np.sqrt((pos[0] - car_center[0])**2 + (pos[1] - car_center[1])**2)
        
        first_dist = dist(positions[0])
        last_dist = dist(positions[-1])
        
        # Hand is moving toward car if distance is decreasing
        return last_dist < first_dist * 0.8
    
    def _check_vehicle_contact(
        self,
        detections: list,
        car_bbox: tuple,
        timestamp: float
    ) -> Optional[ContactEvent]:
        """
        Check if another vehicle is contacting the target car.

        Filters out false positives from:
        - Stationary parked vehicles nearby
        - Brief/flickering detections
        - Repeated alerts for the same vehicle
        """
        current_vehicle_ids = set()

        for det in detections:
            if det.get('class') not in ['car', 'truck', 'motorcycle', 'bicycle']:
                continue

            other_bbox = det['bbox']
            track_id = det.get('track_id', id(det))  # Use object id if no track_id
            current_vehicle_ids.add(track_id)

            overlap = self._calculate_overlap(other_bbox, car_bbox)

            # Skip self-detections: if overlap is very high (>85%), this is likely
            # the target car being detected twice by the model, not another vehicle
            if overlap > 0.85:
                logger.debug(f"Ignoring vehicle detection with {overlap:.0%} overlap (likely self-detection)")
                continue

            # Update or create tracked vehicle
            if track_id in self.tracked_vehicles:
                vehicle = self.tracked_vehicles[track_id]
                vehicle.bbox = other_bbox
                vehicle.bbox_history.append(other_bbox)
                # Keep only last N frames of history
                vehicle.bbox_history = vehicle.bbox_history[-self.vehicle_stationary_frames:]
                vehicle.frames_tracked += 1

                # Update overlap tracking
                if overlap > self.vehicle_overlap_threshold:
                    vehicle.frames_overlapping += 1
                else:
                    vehicle.frames_overlapping = 0

                # Check if vehicle is stationary (hasn't moved significantly)
                vehicle.is_stationary = self._is_vehicle_stationary(vehicle)
            else:
                # New vehicle detected
                self.tracked_vehicles[track_id] = TrackedVehicle(
                    track_id=track_id,
                    bbox=other_bbox,
                    bbox_history=[other_bbox],
                    frames_tracked=1,
                    frames_overlapping=1 if overlap > self.vehicle_overlap_threshold else 0,
                    last_alert_timestamp=0.0,
                    is_stationary=False  # Can't determine yet
                )
                continue  # Need more frames to evaluate

            vehicle = self.tracked_vehicles[track_id]

            # Check all conditions for a valid vehicle contact alert
            if self._should_alert_vehicle_contact(vehicle, overlap, timestamp):
                # Update cooldown timestamp
                vehicle.last_alert_timestamp = timestamp
                # Reset overlap counter to prevent rapid re-alerts
                vehicle.frames_overlapping = 0

                logger.info(
                    f"Vehicle contact detected: track_id={track_id}, "
                    f"overlap={overlap:.1%}, stationary={vehicle.is_stationary}, "
                    f"frames_overlapping={vehicle.frames_tracked}"
                )

                return ContactEvent(
                    contact_type=ContactType.VEHICLE_CONTACT,
                    confidence=min(overlap * 2, 1.0),  # More conservative confidence scaling
                    location=self._get_overlap_center(other_bbox, car_bbox),
                    actor_id=track_id,
                    timestamp=timestamp
                )

        # Clean up vehicles that are no longer detected
        for track_id in list(self.tracked_vehicles.keys()):
            if track_id not in current_vehicle_ids:
                del self.tracked_vehicles[track_id]

        return None

    def _is_vehicle_stationary(self, vehicle: TrackedVehicle) -> bool:
        """
        Determine if a vehicle is stationary based on bbox movement history.

        A vehicle is considered stationary if its bounding box center hasn't
        moved more than the threshold over the tracking period.
        """
        if len(vehicle.bbox_history) < self.vehicle_stationary_frames // 2:
            # Not enough history to determine - assume moving (safer)
            return False

        # Calculate center positions from bbox history
        centers = []
        for bbox in vehicle.bbox_history:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))

        # Check max movement from first position
        first_center = centers[0]
        max_movement = 0.0
        for center in centers[1:]:
            movement = np.sqrt(
                (center[0] - first_center[0]) ** 2 +
                (center[1] - first_center[1]) ** 2
            )
            max_movement = max(max_movement, movement)

        return max_movement < self.vehicle_stationary_threshold

    def _should_alert_vehicle_contact(
        self,
        vehicle: TrackedVehicle,
        overlap: float,
        timestamp: float
    ) -> bool:
        """
        Determine if we should generate an alert for this vehicle contact.

        Requires ALL of:
        1. Sufficient overlap (above threshold)
        2. Persistent overlap (for multiple frames)
        3. Vehicle is NOT stationary (it's actively moving/approaching)
        4. Cooldown period has passed since last alert for this vehicle
        """
        # 1. Check overlap threshold
        if overlap <= self.vehicle_overlap_threshold:
            return False

        # 2. Check persistence (must overlap for multiple consecutive frames)
        if vehicle.frames_overlapping < self.vehicle_persistence_frames:
            logger.debug(
                f"Vehicle {vehicle.track_id}: overlap persistent for "
                f"{vehicle.frames_overlapping}/{self.vehicle_persistence_frames} frames"
            )
            return False

        # 3. Check if vehicle is stationary (parked cars don't trigger alerts)
        if vehicle.is_stationary:
            logger.debug(
                f"Vehicle {vehicle.track_id}: ignoring stationary vehicle "
                f"(likely parked car)"
            )
            return False

        # 4. Check cooldown
        time_since_last_alert = timestamp - vehicle.last_alert_timestamp
        if time_since_last_alert < self.vehicle_cooldown_seconds:
            logger.debug(
                f"Vehicle {vehicle.track_id}: cooldown active "
                f"({time_since_last_alert:.1f}s < {self.vehicle_cooldown_seconds}s)"
            )
            return False

        return True
    
    def _check_impact_event(
        self,
        frame: np.ndarray,
        car_bbox: tuple,
        timestamp: float
    ) -> Optional[ContactEvent]:
        """Detect sudden motion/impact near the car."""
        if self.prev_frame is None:
            return None
        
        cx1, cy1, cx2, cy2 = car_bbox
        margin = 20
        
        # Expand region slightly
        rx1 = max(0, cx1 - margin)
        ry1 = max(0, cy1 - margin)
        rx2 = min(frame.shape[1], cx2 + margin)
        ry2 = min(frame.shape[0], cy2 + margin)
        
        # Extract regions
        current_region = frame[ry1:ry2, rx1:rx2]
        prev_region = self.prev_frame[ry1:ry2, rx1:rx2]
        
        if current_region.shape != prev_region.shape:
            return None
        
        # Calculate frame difference
        diff = cv2.absdiff(current_region, prev_region)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        
        # Count high-motion pixels
        motion_pixels = np.sum(diff_gray > self.motion_threshold)
        motion_ratio = motion_pixels / diff_gray.size
        
        # Sudden significant motion = potential impact
        if motion_ratio > 0.15:  # 15% of pixels changed significantly
            # Find center of motion
            motion_mask = diff_gray > self.motion_threshold
            if np.any(motion_mask):
                ys, xs = np.where(motion_mask)
                center = (int(np.mean(xs)) + rx1, int(np.mean(ys)) + ry1)
            else:
                center = ((cx1 + cx2) // 2, (cy1 + cy2) // 2)
            
            return ContactEvent(
                contact_type=ContactType.IMPACT,
                confidence=min(motion_ratio * 3, 1.0),
                location=center,
                actor_id=None,
                timestamp=timestamp
            )
        
        return None
    
    def _calculate_distance(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate minimum distance between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate horizontal gap
        if x2_1 < x1_2:
            h_gap = x1_2 - x2_1
        elif x2_2 < x1_1:
            h_gap = x1_1 - x2_2
        else:
            h_gap = 0
        
        # Calculate vertical gap
        if y2_1 < y1_2:
            v_gap = y1_2 - y2_1
        elif y2_2 < y1_1:
            v_gap = y1_1 - y2_2
        else:
            v_gap = 0
        
        return np.sqrt(h_gap**2 + v_gap**2)
    
    def _calculate_overlap(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate IoU-style overlap ratio."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return intersection / min(area1, area2)
    
    def _get_overlap_center(self, bbox1: tuple, bbox2: tuple) -> Tuple[int, int]:
        """Get center point of overlapping region."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _extract_hand_positions(self, keypoints) -> List[Tuple[int, int]]:
        """Extract hand positions from pose keypoints."""
        positions = []
        
        for wrist_key in ['left_wrist', 'right_wrist']:
            idx = self.KEYPOINTS.get(wrist_key)
            if idx is None:
                continue
            
            if isinstance(keypoints, dict):
                wrist = keypoints.get(wrist_key)
            elif isinstance(keypoints, (list, np.ndarray)) and len(keypoints) > idx:
                wrist = keypoints[idx]
            else:
                continue
            
            if wrist is not None and len(wrist) >= 2:
                positions.append((int(wrist[0]), int(wrist[1])))
        
        return positions
    
    @property
    def has_active_contact(self) -> bool:
        """Returns True if any contact is currently active."""
        return len(self.active_contacts) > 0
    
    def get_contact_summary(self) -> dict:
        """Get summary of recent contact events."""
        return {
            "active_contacts": len(self.active_contacts),
            "tracked_persons": len(self.tracked_persons),
            "recent_events": len(self.contact_history),
            "contact_types": [e.contact_type.value for e in self.contact_history]
        }


# Need cv2 for frame processing
try:
    import cv2
except ImportError:
    logger.warning("OpenCV not available - some features will be limited")
    cv2 = None
