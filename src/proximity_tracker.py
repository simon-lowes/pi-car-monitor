"""
Proximity Tracker - Stage 2
============================
Detects and tracks objects within proximity of the target car.

This stage:
1. Takes the car bounding box from Stage 1
2. Defines a proximity zone around it
3. Detects people, vehicles, and other objects entering this zone
4. Tracks their movement and dwell time

Only objects within the proximity zone are passed to Stage 3 (Contact Classification).
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """An object being tracked in the proximity zone."""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    first_seen: float
    last_seen: float
    positions: List[Tuple[int, int]] = field(default_factory=list)
    in_proximity: bool = False
    frames_in_proximity: int = 0
    distance_to_car: float = float('inf')

    @property
    def dwell_time(self) -> float:
        """Time spent in proximity zone."""
        return self.last_seen - self.first_seen

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of current bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )


class ProximityTracker:
    """
    Tracks objects near the target car.

    Uses simple IoU-based tracking with proximity zone filtering.
    Objects must enter the proximity zone to be considered for
    contact detection.
    """

    def __init__(self, config: dict, hailo_device):
        self.config = config
        self.hailo = hailo_device

        detection_config = config.get("detection", {})
        self.proximity_buffer = config.get("zones", {}).get("proximity_buffer", 50)
        self.proximity_confidence = detection_config.get("proximity_confidence", 0.6)
        self.trackable_objects = detection_config.get("trackable_objects", [
            "person", "bicycle", "motorcycle", "car", "truck", "dog"
        ])

        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 0

        # Tracking parameters
        self.iou_threshold = 0.3  # For matching detections to tracks
        self.max_age = 30  # Frames before removing lost track
        self.min_hits = 3  # Frames before confirming track

        # Statistics
        self.stats = {
            "total_tracks": 0,
            "proximity_events": 0,
        }

    def process_frame(
        self,
        frame: np.ndarray,
        car_bbox: Optional[Tuple[int, int, int, int]],
        timestamp: float = None
    ) -> List[Dict]:
        """
        Process frame to detect and track objects near the car.

        Args:
            frame: BGR image
            car_bbox: Bounding box of target car (x1, y1, x2, y2)
            timestamp: Frame timestamp

        Returns:
            List of objects in proximity zone (for Stage 3)
        """
        if timestamp is None:
            timestamp = time.time()

        if car_bbox is None:
            # No car detected, clear old tracks
            self._age_tracks(timestamp)
            return []

        # Calculate proximity zone
        proximity_zone = self._calculate_proximity_zone(car_bbox, frame.shape)

        # Run object detection
        detections = self._detect_objects(frame)

        # Filter to trackable objects
        filtered = [d for d in detections if d['class'] in self.trackable_objects]

        # Update tracking
        self._update_tracks(filtered, timestamp)

        # Check which tracks are in proximity
        nearby_objects = []
        for track_id, track in self.tracked_objects.items():
            distance = self._distance_to_car(track.center, car_bbox)
            track.distance_to_car = distance

            if self._is_in_proximity(track.bbox, proximity_zone):
                track.in_proximity = True
                track.frames_in_proximity += 1

                if track.frames_in_proximity >= self.min_hits:
                    nearby_objects.append({
                        'track_id': track.track_id,
                        'class': track.class_name,
                        'bbox': track.bbox,
                        'confidence': track.confidence,
                        'distance_to_car': distance,
                        'dwell_time': track.dwell_time,
                        'frames_in_proximity': track.frames_in_proximity
                    })

                    if track.frames_in_proximity == self.min_hits:
                        self.stats["proximity_events"] += 1
                        logger.info(
                            f"Proximity: {track.class_name} entered zone "
                            f"(track {track.track_id}, dist: {distance:.0f}px)"
                        )
            else:
                track.in_proximity = False
                track.frames_in_proximity = 0

        return nearby_objects

    def _calculate_proximity_zone(
        self,
        car_bbox: Tuple[int, int, int, int],
        frame_shape: tuple
    ) -> Tuple[int, int, int, int]:
        """
        Calculate the proximity zone around the car.

        Expands the car bbox by the proximity buffer.
        """
        x1, y1, x2, y2 = car_bbox
        buffer = self.proximity_buffer

        frame_h, frame_w = frame_shape[:2]

        return (
            max(0, x1 - buffer),
            max(0, y1 - buffer),
            min(frame_w, x2 + buffer),
            min(frame_h, y2 + buffer)
        )

    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on frame."""
        from utils.hailo_utils import postprocess_yolov8_detections

        # Use main detector model
        model_name = "detector"
        outputs = self.hailo.run_inference(model_name, frame)

        if outputs is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        detections = postprocess_yolov8_detections(
            outputs,
            conf_threshold=self.proximity_confidence,
            orig_shape=(frame_h, frame_w)
        )

        return detections

    def _update_tracks(self, detections: List[Dict], timestamp: float):
        """Update tracks with new detections using IoU matching."""
        # Age existing tracks
        self._age_tracks(timestamp)

        if not detections:
            return

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)

        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            track = self.tracked_objects[track_id]
            track.bbox = det['bbox']
            track.confidence = det['confidence']
            track.last_seen = timestamp
            track.positions.append(track.center)
            # Keep only last 30 positions
            track.positions = track.positions[-30:]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det, timestamp)

        # Remove old unmatched tracks (handled in _age_tracks)

    def _match_detections(
        self,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.

        Returns:
            matched: List of (track_id, detection_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_tracks: List of unmatched track IDs
        """
        if not self.tracked_objects:
            return [], list(range(len(detections))), []

        # Calculate IoU matrix
        track_ids = list(self.tracked_objects.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracked_objects[track_id].bbox
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track_bbox, det['bbox'])

        # Greedy matching
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(track_ids)

        while True:
            if iou_matrix.size == 0:
                break

            # Find best match
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_id = track_ids[i]

            matched.append((track_id, j))
            unmatched_dets.remove(j)
            unmatched_tracks.remove(track_id)

            # Remove matched row and column
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        return matched, unmatched_dets, unmatched_tracks

    def _create_track(self, detection: Dict, timestamp: float):
        """Create a new track for an unmatched detection."""
        track = TrackedObject(
            track_id=self.next_track_id,
            class_name=detection['class'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            first_seen=timestamp,
            last_seen=timestamp,
            positions=[self._bbox_center(detection['bbox'])]
        )

        self.tracked_objects[self.next_track_id] = track
        self.next_track_id += 1
        self.stats["total_tracks"] += 1

    def _age_tracks(self, timestamp: float):
        """Remove tracks that haven't been seen recently."""
        # Calculate age threshold (in seconds)
        max_age_seconds = self.max_age / 15.0  # Assuming ~15 fps

        to_remove = []
        for track_id, track in self.tracked_objects.items():
            age = timestamp - track.last_seen
            if age > max_age_seconds:
                to_remove.append(track_id)

        for track_id in to_remove:
            if self.tracked_objects[track_id].in_proximity:
                logger.debug(f"Track {track_id} left proximity zone")
            del self.tracked_objects[track_id]

    def _is_in_proximity(
        self,
        obj_bbox: Tuple[int, int, int, int],
        proximity_zone: Tuple[int, int, int, int]
    ) -> bool:
        """Check if object bbox overlaps with proximity zone."""
        # Check if any part of object is in proximity zone
        x1 = max(obj_bbox[0], proximity_zone[0])
        y1 = max(obj_bbox[1], proximity_zone[1])
        x2 = min(obj_bbox[2], proximity_zone[2])
        y2 = min(obj_bbox[3], proximity_zone[3])

        return x2 > x1 and y2 > y1

    def _distance_to_car(
        self,
        point: Tuple[int, int],
        car_bbox: Tuple[int, int, int, int]
    ) -> float:
        """Calculate minimum distance from point to car bbox edge."""
        px, py = point
        cx1, cy1, cx2, cy2 = car_bbox

        # Find nearest point on car bbox
        nearest_x = max(cx1, min(px, cx2))
        nearest_y = max(cy1, min(py, cy2))

        # Calculate distance
        dx = px - nearest_x
        dy = py - nearest_y

        return np.sqrt(dx * dx + dy * dy)

    def _iou(self, box1: tuple, box2: tuple) -> float:
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

    def _bbox_center(self, bbox: tuple) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (
            (bbox[0] + bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2
        )

    def get_stats(self) -> dict:
        """Get tracking statistics."""
        return {
            **self.stats,
            "active_tracks": len(self.tracked_objects),
            "tracks_in_proximity": sum(
                1 for t in self.tracked_objects.values() if t.in_proximity
            )
        }

    def clear(self):
        """Clear all tracking state."""
        self.tracked_objects.clear()
        self.next_track_id = 0
