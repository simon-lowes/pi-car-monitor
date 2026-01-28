"""
Owner Profile Module
====================
Stores and matches owner appearance data to suppress false alerts.

The system learns the owner's appearance when they reply "me" to alerts,
and uses this to recognize them in future detections.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def extract_person_features(frame: np.ndarray, person_bbox: tuple) -> Optional[np.ndarray]:
    """
    Extract appearance features from a person in the frame.

    Uses color histograms and simple texture descriptors to create
    a feature vector that can be compared via cosine similarity.

    Args:
        frame: The full frame (BGR image)
        person_bbox: (x1, y1, x2, y2) bounding box of the person

    Returns:
        Normalized feature vector (np.ndarray) or None if extraction fails
    """
    if not CV2_AVAILABLE or frame is None:
        return None

    try:
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        h, w = frame.shape[:2]

        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract person crop
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.size == 0:
            return None

        # Resize to standard size for consistent features
        standard_size = (64, 128)  # width, height
        resized = cv2.resize(person_crop, standard_size)

        # Convert to HSV for color features
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Extract histograms
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])  # Hue: 16 bins
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])   # Sat: 8 bins
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])   # Val: 8 bins

        # Extract grayscale texture features (simple gradients)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude histogram
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        mag_hist, _ = np.histogram(magnitude.flatten(), bins=16, range=(0, 255))

        # Combine all features
        features = np.concatenate([
            h_hist.flatten(),      # 16 values
            s_hist.flatten(),      # 8 values
            v_hist.flatten(),      # 8 values
            mag_hist.astype(float) # 16 values
        ])  # Total: 48 features

        # Normalize to unit vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features.astype(np.float32)

    except Exception as e:
        logger.warning(f"Feature extraction failed: {e}")
        return None


class OwnerProfile:
    """
    Manages owner appearance profile for recognition.

    Stores feature vectors extracted from frames where the owner
    confirmed their presence. Uses simple cosine similarity for matching.
    """

    def __init__(
        self,
        profile_path: str,
        confidence_threshold: float = 0.7,
        max_samples: int = 50
    ):
        self.profile_path = Path(profile_path)
        self.confidence_threshold = confidence_threshold
        self.max_samples = max_samples

        # Stored samples: list of (timestamp, features) tuples
        self.samples: List[Tuple[float, np.ndarray]] = []

        # Event tracking for reply correlation
        self.recent_events: Dict[int, dict] = {}  # event_id -> event_data
        self._next_event_id = 1

        # Load existing profile
        self.load()

    def create_event(
        self,
        recording_path: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        features: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Create a new event for tracking.

        Returns event_id that can be included in alert messages.
        """
        event_id = self._next_event_id
        self._next_event_id += 1

        self.recent_events[event_id] = {
            'id': event_id,
            'timestamp': timestamp or datetime.now().timestamp(),
            'recording_path': recording_path,
            'snapshot_path': snapshot_path,
            'features': features,
            'confirmed_owner': False
        }

        # Keep only last 20 events to avoid memory bloat
        if len(self.recent_events) > 20:
            oldest_id = min(self.recent_events.keys())
            del self.recent_events[oldest_id]

        logger.debug(f"Created event {event_id}")
        return event_id

    def get_event(self, event_id: int) -> Optional[dict]:
        """Get event data by ID."""
        return self.recent_events.get(event_id)

    def get_latest_event(self) -> Optional[dict]:
        """Get the most recent event."""
        if not self.recent_events:
            return None
        latest_id = max(self.recent_events.keys())
        return self.recent_events[latest_id]

    def confirm_owner(
        self,
        event_id: Optional[int] = None,
        features: Optional[np.ndarray] = None
    ) -> bool:
        """
        Confirm that an event was the owner.

        If event_id is None, uses the most recent event.
        Adds the features to the owner profile.
        """
        if event_id is None:
            event = self.get_latest_event()
            if event:
                event_id = event['id']

        if event_id is None:
            logger.warning("No event to confirm")
            return False

        event = self.get_event(event_id)
        if event is None:
            logger.warning(f"Event {event_id} not found")
            return False

        # Use provided features or event features
        sample_features = features if features is not None else event.get('features')

        if sample_features is not None:
            self.add_sample(sample_features)
            event['confirmed_owner'] = True
            logger.info(f"Event {event_id} confirmed as owner, sample added")
            return True
        else:
            # No features available, just mark as confirmed
            event['confirmed_owner'] = True
            logger.info(f"Event {event_id} confirmed as owner (no features to store)")
            return True

    def add_sample(self, features: np.ndarray, timestamp: Optional[float] = None):
        """
        Add a feature sample to the owner profile.

        Args:
            features: Feature vector (numpy array)
            timestamp: When this sample was captured
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        self.samples.append((timestamp, features))

        # Trim old samples if exceeding max
        if len(self.samples) > self.max_samples:
            # Remove oldest samples
            self.samples = self.samples[-self.max_samples:]

        # Save after adding
        self.save()

        logger.info(f"Owner sample added (total: {len(self.samples)})")

    def match(self, features: np.ndarray) -> float:
        """
        Check if features match the owner profile.

        Args:
            features: Feature vector to match

        Returns:
            Confidence score (0-1) that this is the owner
        """
        if not self.samples:
            return 0.0

        # Normalize input features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        # Compare against all samples using cosine similarity
        similarities = []
        for _, sample_features in self.samples:
            sim = np.dot(features, sample_features)
            similarities.append(sim)

        # Return max similarity (best match)
        # Could also use mean or weighted average
        max_sim = max(similarities)

        # Convert to 0-1 range (cosine sim is -1 to 1)
        confidence = (max_sim + 1) / 2

        return float(confidence)

    def is_owner(self, features: np.ndarray) -> bool:
        """
        Check if features likely belong to owner.

        Args:
            features: Feature vector to check

        Returns:
            True if confidence exceeds threshold
        """
        confidence = self.match(features)
        return confidence >= self.confidence_threshold

    def save(self):
        """Save profile to disk."""
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'version': 1,
                'samples': [(ts, feat.tolist()) for ts, feat in self.samples],
                'next_event_id': self._next_event_id
            }

            with open(self.profile_path, 'w') as f:
                json.dump(data, f)

            logger.debug(f"Profile saved: {len(self.samples)} samples")

        except Exception as e:
            logger.error(f"Failed to save profile: {e}")

    def load(self):
        """Load profile from disk."""
        if not self.profile_path.exists():
            logger.info("No existing owner profile found")
            return

        try:
            with open(self.profile_path, 'r') as f:
                data = json.load(f)

            self.samples = [
                (ts, np.array(feat, dtype=np.float32))
                for ts, feat in data.get('samples', [])
            ]
            self._next_event_id = data.get('next_event_id', 1)

            logger.info(f"Owner profile loaded: {len(self.samples)} samples")

        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            self.samples = []

    def reset(self):
        """Clear all owner data."""
        self.samples = []
        self.recent_events = {}

        if self.profile_path.exists():
            self.profile_path.unlink()

        logger.info("Owner profile reset")

    @property
    def sample_count(self) -> int:
        """Number of stored owner samples."""
        return len(self.samples)

    @property
    def is_trained(self) -> bool:
        """Whether profile has enough samples for recognition."""
        return len(self.samples) >= 3


def create_profile_from_config(config: dict) -> Optional[OwnerProfile]:
    """
    Create OwnerProfile from configuration.

    Args:
        config: Configuration dict with owner_recognition section

    Returns:
        OwnerProfile instance or None if disabled
    """
    owner_config = config.get('owner_recognition', {})

    if not owner_config.get('enabled', True):
        logger.info("Owner recognition disabled in config")
        return None

    profile_path = owner_config.get(
        'profile_path',
        '/home/PiAi/pi-car-monitor/config/.owner_profile'
    )

    confidence_threshold = owner_config.get('confidence_threshold', 0.7)

    return OwnerProfile(
        profile_path=profile_path,
        confidence_threshold=confidence_threshold
    )
