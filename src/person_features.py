"""
Person Feature Extractor
========================
Extracts appearance features from person detections for owner recognition.

Features extracted:
1. Color histograms (upper/lower body clothing)
2. Body proportions from pose keypoints
3. Texture patterns (optional)
4. Height estimation relative to known objects

These features are used to build an owner profile and recognize
the owner during departure events.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# COCO pose keypoint indices
KEYPOINT_NOSE = 0
KEYPOINT_LEFT_EYE = 1
KEYPOINT_RIGHT_EYE = 2
KEYPOINT_LEFT_EAR = 3
KEYPOINT_RIGHT_EAR = 4
KEYPOINT_LEFT_SHOULDER = 5
KEYPOINT_RIGHT_SHOULDER = 6
KEYPOINT_LEFT_ELBOW = 7
KEYPOINT_RIGHT_ELBOW = 8
KEYPOINT_LEFT_WRIST = 9
KEYPOINT_RIGHT_WRIST = 10
KEYPOINT_LEFT_HIP = 11
KEYPOINT_RIGHT_HIP = 12
KEYPOINT_LEFT_KNEE = 13
KEYPOINT_RIGHT_KNEE = 14
KEYPOINT_LEFT_ANKLE = 15
KEYPOINT_RIGHT_ANKLE = 16


@dataclass
class PersonFeatures:
    """Extracted features for a person."""
    # Color features
    upper_body_histogram: np.ndarray  # HSV histogram of upper body
    lower_body_histogram: np.ndarray  # HSV histogram of lower body
    dominant_colors: List[Tuple[int, int, int]]  # Top 3 BGR colors

    # Body proportion features (normalized)
    shoulder_width_ratio: Optional[float] = None  # Shoulder width / height
    torso_ratio: Optional[float] = None  # Torso length / total height
    leg_ratio: Optional[float] = None  # Leg length / total height

    # Size features
    bbox_aspect_ratio: float = 0.0  # Height / width of bbox
    estimated_height_px: int = 0  # Height in pixels

    # Combined feature vector for matching
    feature_vector: Optional[np.ndarray] = None

    # Metadata
    confidence: float = 0.0
    timestamp: float = 0.0
    source_bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class DepartureActor:
    """Person detected during a departure event."""
    person_id: int  # Track ID if available
    bbox: Tuple[int, int, int, int]
    features: Optional[PersonFeatures]
    pose_keypoints: Optional[np.ndarray]  # 17x3 array (x, y, confidence)

    # Relation to car
    overlap_with_car: float  # IoU with car bbox
    position: str  # 'entering', 'near', 'inside'
    distance_to_car: float  # Pixels from car center

    # Tracking
    frames_visible: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0

    # Frame data for later training
    snapshot: Optional[np.ndarray] = None  # Cropped image


class PersonFeatureExtractor:
    """
    Extracts appearance features from person detections.

    Uses a combination of color histograms, body proportions,
    and spatial features to create a recognizable profile.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Histogram settings
        self.h_bins = 18  # Hue bins (0-180)
        self.s_bins = 8   # Saturation bins
        self.v_bins = 8   # Value bins

        # Feature vector size
        self.color_feature_size = self.h_bins * self.s_bins  # 144
        self.proportion_feature_size = 6
        self.total_feature_size = (
            self.color_feature_size * 2 +  # Upper + lower body
            self.proportion_feature_size +
            4  # Additional features
        )

        logger.info(f"PersonFeatureExtractor initialized, feature size={self.total_feature_size}")

    def extract_features(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        pose_keypoints: Optional[np.ndarray] = None,
        timestamp: float = 0.0
    ) -> PersonFeatures:
        """
        Extract features from a person detection.

        Args:
            frame: Full frame image (BGR)
            person_bbox: Bounding box (x1, y1, x2, y2)
            pose_keypoints: Optional 17x3 array of keypoints
            timestamp: When this was captured

        Returns:
            PersonFeatures with extracted data
        """
        x1, y1, x2, y2 = person_bbox

        # Validate and clip bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return self._empty_features(timestamp)

        person_crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = person_crop.shape[:2]

        # Split into upper and lower body regions
        if pose_keypoints is not None and len(pose_keypoints) >= 17:
            upper_region, lower_region = self._split_by_pose(
                person_crop, pose_keypoints, (x1, y1)
            )
        else:
            # Simple split at 40% from top (approximate waist)
            split_y = int(crop_h * 0.4)
            upper_region = person_crop[:split_y, :]
            lower_region = person_crop[split_y:, :]

        # Extract color histograms
        upper_hist = self._extract_color_histogram(upper_region)
        lower_hist = self._extract_color_histogram(lower_region)

        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(person_crop, k=3)

        # Extract body proportions from pose
        proportions = self._extract_proportions(pose_keypoints, crop_h)

        # Calculate bbox features
        bbox_aspect = crop_h / max(crop_w, 1)

        # Build combined feature vector
        feature_vector = self._build_feature_vector(
            upper_hist, lower_hist, proportions, bbox_aspect
        )

        # Calculate confidence based on feature quality
        confidence = self._calculate_confidence(
            upper_region, lower_region, pose_keypoints
        )

        return PersonFeatures(
            upper_body_histogram=upper_hist,
            lower_body_histogram=lower_hist,
            dominant_colors=dominant_colors,
            shoulder_width_ratio=proportions.get('shoulder_width_ratio'),
            torso_ratio=proportions.get('torso_ratio'),
            leg_ratio=proportions.get('leg_ratio'),
            bbox_aspect_ratio=bbox_aspect,
            estimated_height_px=crop_h,
            feature_vector=feature_vector,
            confidence=confidence,
            timestamp=timestamp,
            source_bbox=person_bbox
        )

    def _split_by_pose(
        self,
        crop: np.ndarray,
        keypoints: np.ndarray,
        offset: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split person crop into upper/lower body using pose keypoints."""
        crop_h, crop_w = crop.shape[:2]
        ox, oy = offset

        # Try to find hip midpoint
        left_hip = keypoints[KEYPOINT_LEFT_HIP]
        right_hip = keypoints[KEYPOINT_RIGHT_HIP]

        if left_hip[2] > 0.3 and right_hip[2] > 0.3:
            # Use hip y-coordinate as split point
            hip_y = int((left_hip[1] + right_hip[1]) / 2) - oy
            hip_y = max(int(crop_h * 0.3), min(int(crop_h * 0.6), hip_y))
        else:
            # Fall back to shoulders if hips not visible
            left_shoulder = keypoints[KEYPOINT_LEFT_SHOULDER]
            right_shoulder = keypoints[KEYPOINT_RIGHT_SHOULDER]

            if left_shoulder[2] > 0.3 and right_shoulder[2] > 0.3:
                shoulder_y = int((left_shoulder[1] + right_shoulder[1]) / 2) - oy
                # Estimate hip as 1.2x shoulder distance from top
                hip_y = int(shoulder_y * 1.5)
                hip_y = max(int(crop_h * 0.3), min(int(crop_h * 0.6), hip_y))
            else:
                # Default split
                hip_y = int(crop_h * 0.4)

        upper = crop[:hip_y, :]
        lower = crop[hip_y:, :]

        return upper, lower

    def _extract_color_histogram(self, region: np.ndarray) -> np.ndarray:
        """Extract normalized HSV color histogram from a region."""
        if region.size == 0:
            return np.zeros(self.h_bins * self.s_bins, dtype=np.float32)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Calculate 2D histogram of H and S (ignore V for lighting invariance)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.h_bins, self.s_bins],
            [0, 180, 0, 256]
        )

        # Normalize
        hist = hist.flatten()
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum

        return hist.astype(np.float32)

    def _extract_dominant_colors(
        self,
        region: np.ndarray,
        k: int = 3
    ) -> List[Tuple[int, int, int]]:
        """Extract k dominant colors using k-means clustering."""
        if region.size == 0:
            return [(0, 0, 0)] * k

        # Reshape to list of pixels
        pixels = region.reshape(-1, 3).astype(np.float32)

        # Skip if too few pixels
        if len(pixels) < k:
            return [(0, 0, 0)] * k

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )

            # Sort by cluster size (most common first)
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(-counts)

            colors = []
            for idx in sorted_indices[:k]:
                color = tuple(int(c) for c in centers[idx])
                colors.append(color)

            return colors

        except Exception as e:
            logger.debug(f"K-means failed: {e}")
            return [(0, 0, 0)] * k

    def _extract_proportions(
        self,
        keypoints: Optional[np.ndarray],
        crop_height: int
    ) -> Dict[str, Optional[float]]:
        """Extract body proportion features from pose keypoints."""
        result = {
            'shoulder_width_ratio': None,
            'torso_ratio': None,
            'leg_ratio': None,
            'arm_ratio': None,
            'head_ratio': None,
            'pose_confidence': 0.0
        }

        if keypoints is None or len(keypoints) < 17:
            return result

        # Calculate various proportions if keypoints are available
        try:
            # Shoulder width
            ls = keypoints[KEYPOINT_LEFT_SHOULDER]
            rs = keypoints[KEYPOINT_RIGHT_SHOULDER]
            if ls[2] > 0.3 and rs[2] > 0.3:
                shoulder_width = abs(ls[0] - rs[0])
                result['shoulder_width_ratio'] = shoulder_width / max(crop_height, 1)

            # Torso length (shoulder to hip)
            lh = keypoints[KEYPOINT_LEFT_HIP]
            rh = keypoints[KEYPOINT_RIGHT_HIP]
            if ls[2] > 0.3 and rs[2] > 0.3 and lh[2] > 0.3 and rh[2] > 0.3:
                shoulder_y = (ls[1] + rs[1]) / 2
                hip_y = (lh[1] + rh[1]) / 2
                torso_length = abs(hip_y - shoulder_y)
                result['torso_ratio'] = torso_length / max(crop_height, 1)

            # Leg length (hip to ankle)
            la = keypoints[KEYPOINT_LEFT_ANKLE]
            ra = keypoints[KEYPOINT_RIGHT_ANKLE]
            if lh[2] > 0.3 and la[2] > 0.3:
                leg_length = abs(la[1] - lh[1])
                result['leg_ratio'] = leg_length / max(crop_height, 1)
            elif rh[2] > 0.3 and ra[2] > 0.3:
                leg_length = abs(ra[1] - rh[1])
                result['leg_ratio'] = leg_length / max(crop_height, 1)

            # Head size (nose to shoulder)
            nose = keypoints[KEYPOINT_NOSE]
            if nose[2] > 0.3 and (ls[2] > 0.3 or rs[2] > 0.3):
                shoulder_y = ls[1] if ls[2] > 0.3 else rs[1]
                head_length = abs(shoulder_y - nose[1])
                result['head_ratio'] = head_length / max(crop_height, 1)

            # Overall pose confidence
            valid_keypoints = sum(1 for kp in keypoints if kp[2] > 0.3)
            result['pose_confidence'] = valid_keypoints / 17.0

        except Exception as e:
            logger.debug(f"Proportion extraction failed: {e}")

        return result

    def _build_feature_vector(
        self,
        upper_hist: np.ndarray,
        lower_hist: np.ndarray,
        proportions: Dict[str, Optional[float]],
        bbox_aspect: float
    ) -> np.ndarray:
        """Build combined feature vector for matching."""
        features = []

        # Color features (already normalized histograms)
        features.extend(upper_hist.flatten())
        features.extend(lower_hist.flatten())

        # Proportion features (normalized to 0-1 range, use -1 for missing)
        prop_keys = ['shoulder_width_ratio', 'torso_ratio', 'leg_ratio',
                     'arm_ratio', 'head_ratio', 'pose_confidence']
        for key in prop_keys:
            val = proportions.get(key)
            if val is not None:
                features.append(float(val))
            else:
                features.append(-1.0)  # Missing indicator

        # Bbox features
        features.append(min(bbox_aspect / 3.0, 1.0))  # Normalize aspect ratio
        features.append(0.0)  # Placeholder for height relative to car
        features.append(0.0)  # Placeholder
        features.append(0.0)  # Placeholder

        return np.array(features, dtype=np.float32)

    def _calculate_confidence(
        self,
        upper_region: np.ndarray,
        lower_region: np.ndarray,
        keypoints: Optional[np.ndarray]
    ) -> float:
        """Calculate confidence in feature quality."""
        confidence = 0.5  # Base confidence

        # Boost for good region sizes
        if upper_region.size > 1000:
            confidence += 0.1
        if lower_region.size > 1000:
            confidence += 0.1

        # Boost for pose data
        if keypoints is not None:
            valid_kps = sum(1 for kp in keypoints if kp[2] > 0.3)
            confidence += (valid_kps / 17) * 0.3

        return min(confidence, 1.0)

    def _empty_features(self, timestamp: float) -> PersonFeatures:
        """Return empty features when extraction fails."""
        return PersonFeatures(
            upper_body_histogram=np.zeros(self.h_bins * self.s_bins, dtype=np.float32),
            lower_body_histogram=np.zeros(self.h_bins * self.s_bins, dtype=np.float32),
            dominant_colors=[(0, 0, 0), (0, 0, 0), (0, 0, 0)],
            feature_vector=np.zeros(self.total_feature_size, dtype=np.float32),
            confidence=0.0,
            timestamp=timestamp
        )

    def compare_features(
        self,
        features1: PersonFeatures,
        features2: PersonFeatures
    ) -> float:
        """
        Compare two feature sets and return similarity score.

        Args:
            features1: First person features
            features2: Second person features

        Returns:
            Similarity score 0-1 (1 = identical)
        """
        if features1.feature_vector is None or features2.feature_vector is None:
            return 0.0

        # Cosine similarity on feature vectors
        v1 = features1.feature_vector
        v2 = features2.feature_vector

        # Handle missing values (-1)
        mask = (v1 >= 0) & (v2 >= 0)
        if not np.any(mask):
            return 0.0

        v1_masked = v1[mask]
        v2_masked = v2[mask]

        norm1 = np.linalg.norm(v1_masked)
        norm2 = np.linalg.norm(v2_masked)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(v1_masked, v2_masked) / (norm1 * norm2)

        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2


class DepartureActorTracker:
    """
    Tracks people during departure events to identify who is getting in the car.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.feature_extractor = PersonFeatureExtractor(config)

        # Active tracking
        self.active_actors: Dict[int, DepartureActor] = {}
        self._next_actor_id = 1

        # Departure event data
        self.current_departure_actors: List[DepartureActor] = []

    def process_frame(
        self,
        frame: np.ndarray,
        person_detections: List[Dict[str, Any]],
        pose_detections: Optional[List[Dict[str, Any]]],
        car_bbox: Tuple[int, int, int, int],
        timestamp: float
    ) -> List[DepartureActor]:
        """
        Process a frame during departure to track actors.

        Args:
            frame: Current frame
            person_detections: List of person detections with 'bbox'
            pose_detections: List of pose detections with 'bbox' and 'keypoints'
            car_bbox: Car bounding box
            timestamp: Current timestamp

        Returns:
            List of DepartureActors near the car
        """
        car_x1, car_y1, car_x2, car_y2 = car_bbox
        car_center = ((car_x1 + car_x2) // 2, (car_y1 + car_y2) // 2)

        departure_actors = []

        for det in person_detections:
            person_bbox = det.get('bbox')
            if person_bbox is None:
                continue

            px1, py1, px2, py2 = person_bbox

            # Calculate overlap with car
            overlap = self._calculate_iou(person_bbox, car_bbox)

            # Calculate distance to car center
            person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            distance = np.sqrt(
                (person_center[0] - car_center[0]) ** 2 +
                (person_center[1] - car_center[1]) ** 2
            )

            # Check if person is relevant (near or overlapping car)
            # Expand car zone for "near" detection
            expanded_margin = 100
            near_car = (
                px1 < car_x2 + expanded_margin and
                px2 > car_x1 - expanded_margin and
                py1 < car_y2 + expanded_margin and
                py2 > car_y1 - expanded_margin
            )

            if not near_car and overlap < 0.05:
                continue

            # Determine position
            if overlap > 0.5:
                position = 'inside'
            elif overlap > 0.1:
                position = 'entering'
            else:
                position = 'near'

            # Find matching pose data
            keypoints = None
            if pose_detections:
                keypoints = self._match_pose_to_person(person_bbox, pose_detections)

            # Extract features
            features = self.feature_extractor.extract_features(
                frame, person_bbox, keypoints, timestamp
            )

            # Create snapshot
            h, w = frame.shape[:2]
            snap_x1 = max(0, px1 - 10)
            snap_y1 = max(0, py1 - 10)
            snap_x2 = min(w, px2 + 10)
            snap_y2 = min(h, py2 + 10)
            snapshot = frame[snap_y1:snap_y2, snap_x1:snap_x2].copy()

            # Create or update actor
            actor = DepartureActor(
                person_id=self._next_actor_id,
                bbox=person_bbox,
                features=features,
                pose_keypoints=keypoints,
                overlap_with_car=overlap,
                position=position,
                distance_to_car=distance,
                frames_visible=1,
                first_seen=timestamp,
                last_seen=timestamp,
                snapshot=snapshot
            )
            self._next_actor_id += 1

            departure_actors.append(actor)

        self.current_departure_actors = departure_actors
        return departure_actors

    def _calculate_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _match_pose_to_person(
        self,
        person_bbox: Tuple[int, int, int, int],
        pose_detections: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Find pose detection that best matches a person bbox."""
        best_match = None
        best_iou = 0.3  # Minimum threshold

        for pose in pose_detections:
            pose_bbox = pose.get('bbox')
            if pose_bbox is None:
                continue

            iou = self._calculate_iou(person_bbox, pose_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = pose.get('keypoints')

        return best_match

    def get_primary_actor(self) -> Optional[DepartureActor]:
        """
        Get the most likely person getting into the car.

        Returns the actor with highest overlap/entering position.
        """
        if not self.current_departure_actors:
            return None

        # Sort by position priority and overlap
        position_priority = {'inside': 3, 'entering': 2, 'near': 1}

        sorted_actors = sorted(
            self.current_departure_actors,
            key=lambda a: (
                position_priority.get(a.position, 0),
                a.overlap_with_car,
                -a.distance_to_car
            ),
            reverse=True
        )

        return sorted_actors[0] if sorted_actors else None

    def clear(self):
        """Clear current departure tracking."""
        self.current_departure_actors = []
        self.active_actors = {}
