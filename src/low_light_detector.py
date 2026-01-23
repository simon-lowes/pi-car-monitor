"""
Low-Light Departure Detector
=============================
Specialized detection for car departure in low-light and night conditions.

When standard object detection fails due to darkness, this module detects:
1. Vehicle lights (tail, brake, reverse, headlights, interior)
2. Motion patterns in the known car zone
3. Motion direction (backing out vs stationary)

Works in conjunction with PresenceTracker to detect departures at night.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LightCondition(Enum):
    """Ambient light conditions."""
    DAYLIGHT = auto()      # Bright, full object detection works
    TWILIGHT = auto()      # Transitional, detection may be degraded
    LOW_LIGHT = auto()     # Dark but some visibility
    NIGHT = auto()         # Very dark, rely on lights/motion only


class VehicleLightType(Enum):
    """Types of vehicle lights we can detect."""
    TAIL_LIGHT = auto()      # Red, always on when car running
    BRAKE_LIGHT = auto()     # Bright red, indicates braking
    REVERSE_LIGHT = auto()   # White, car in reverse
    HEADLIGHT = auto()       # White/yellow, front of car
    INTERIOR = auto()        # Dome light, door opened
    INDICATOR = auto()       # Amber/orange, turn signal


@dataclass
class DetectedLight:
    """A detected vehicle light."""
    light_type: VehicleLightType
    location: Tuple[int, int]  # Center point
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    intensity: float  # 0-1, relative brightness
    area: int  # Pixel area
    confidence: float  # Detection confidence
    timestamp: float


@dataclass
class LightPattern:
    """Pattern of lights suggesting vehicle state."""
    pattern_type: str  # 'parked', 'running', 'reversing', 'departing'
    lights: List[DetectedLight]
    confidence: float
    timestamp: float


@dataclass
class MotionAnalysis:
    """Analysis of motion in the car zone."""
    motion_detected: bool
    motion_ratio: float  # Fraction of zone with motion
    motion_center: Optional[Tuple[int, int]]
    motion_direction: Optional[str]  # 'toward_camera', 'away_from_camera', 'lateral', 'stationary'
    flow_magnitude: float  # Average optical flow magnitude
    is_significant: bool  # Large enough to be car movement
    is_car_sized: bool  # Motion area suggests car-sized object
    timestamp: float


class LowLightDetector:
    """
    Detects vehicle departure signals in low-light conditions.

    Uses a combination of:
    - Color-based light detection (tail lights, reverse lights, etc.)
    - Frame differencing for motion
    - Optical flow for motion direction
    - Temporal analysis for light persistence
    """

    def __init__(self, config: dict):
        self.config = config.get('presence_tracking', {})

        # Light detection settings
        self.min_light_area = self.config.get('tail_light_min_area', 50)
        self.light_persistence_frames = self.config.get('light_persistence_frames', 3)

        # Motion detection settings
        self.motion_threshold = self.config.get('motion_threshold', 0.02)
        self.significant_motion_threshold = self.config.get('significant_motion_threshold', 0.10)

        # Frame history for temporal analysis
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_history: deque = deque(maxlen=10)

        # Light detection history
        self._light_history: deque = deque(maxlen=30)  # ~2 seconds at 15fps
        self._persistent_lights: Dict[str, List[DetectedLight]] = {}

        # Motion history
        self._motion_history: deque = deque(maxlen=10)

        # Light condition tracking
        self._brightness_history: deque = deque(maxlen=30)
        self._current_light_condition = LightCondition.DAYLIGHT

        # Time-based hints
        self._last_twilight_check = 0

        logger.info("LowLightDetector initialized")

    def detect_light_condition(
        self,
        frame: np.ndarray,
        use_time_hint: bool = True
    ) -> LightCondition:
        """
        Determine current ambient light conditions.

        Uses histogram analysis for more accurate brightness estimation
        than simple mean, plus optional time-of-day hints.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram-based brightness analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize

        # Calculate weighted mean (histogram mean)
        brightness_values = np.arange(256)
        mean_brightness = np.sum(brightness_values * hist)

        # Calculate brightness percentiles
        cumsum = np.cumsum(hist)
        p10 = np.searchsorted(cumsum, 0.10)  # 10th percentile
        p50 = np.searchsorted(cumsum, 0.50)  # Median
        p90 = np.searchsorted(cumsum, 0.90)  # 90th percentile

        # Dynamic range (contrast)
        dynamic_range = p90 - p10

        # Store for temporal smoothing
        self._brightness_history.append({
            'mean': mean_brightness,
            'median': p50,
            'p10': p10,
            'p90': p90,
            'dynamic_range': dynamic_range,
            'timestamp': time.time()
        })

        # Use smoothed values if we have history
        if len(self._brightness_history) >= 5:
            recent = list(self._brightness_history)[-5:]
            mean_brightness = np.mean([b['mean'] for b in recent])
            p50 = np.mean([b['median'] for b in recent])

        # Classify light condition
        # High dynamic range in dark = artificial lights present
        if mean_brightness > 100 and p50 > 80:
            condition = LightCondition.DAYLIGHT
        elif mean_brightness > 60 or (mean_brightness > 40 and dynamic_range > 100):
            condition = LightCondition.TWILIGHT
        elif mean_brightness > 25 or dynamic_range > 80:
            condition = LightCondition.LOW_LIGHT
        else:
            condition = LightCondition.NIGHT

        # Time-based hint (optional)
        if use_time_hint:
            current_hour = datetime.now().hour
            # Adjust based on typical UK times (where this system is deployed)
            if 6 <= current_hour <= 8 or 17 <= current_hour <= 20:
                # Twilight hours - be more sensitive
                if condition == LightCondition.DAYLIGHT and mean_brightness < 120:
                    condition = LightCondition.TWILIGHT
            elif current_hour < 6 or current_hour > 21:
                # Night hours - shouldn't be daylight
                if condition == LightCondition.DAYLIGHT and mean_brightness < 150:
                    condition = LightCondition.TWILIGHT

        self._current_light_condition = condition
        return condition

    def detect_vehicle_lights(
        self,
        frame: np.ndarray,
        car_zone: Tuple[int, int, int, int]
    ) -> List[DetectedLight]:
        """
        Detect vehicle lights within the car zone.

        Enhanced detection with:
        - Better color filtering
        - Paired light detection
        - Intensity-based classification (tail vs brake)
        - False positive reduction
        """
        x1, y1, x2, y2 = car_zone

        # Validate bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return []

        zone = frame[y1:y2, x1:x2]
        zone_hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        zone_h, zone_w = zone.shape[:2]

        detected_lights = []
        timestamp = time.time()

        # --- Red lights (tail/brake) ---
        # Tail lights: Red with moderate brightness
        # Brake lights: Red with high brightness
        red_mask_low = cv2.inRange(zone_hsv, (0, 80, 80), (10, 255, 255))
        red_mask_high = cv2.inRange(zone_hsv, (170, 80, 80), (180, 255, 255))
        red_mask = red_mask_low | red_mask_high

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_lights = self._extract_lights_from_mask(
            red_mask, zone, zone_hsv, x1, y1, 'red', timestamp
        )
        detected_lights.extend(red_lights)

        # --- White lights (reverse/headlight) ---
        # High value, low saturation
        white_mask = cv2.inRange(zone_hsv, (0, 0, 200), (180, 50, 255))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        white_lights = self._extract_lights_from_mask(
            white_mask, zone, zone_hsv, x1, y1, 'white', timestamp
        )
        detected_lights.extend(white_lights)

        # --- Amber lights (indicators) ---
        amber_mask = cv2.inRange(zone_hsv, (10, 100, 100), (25, 255, 255))
        amber_mask = cv2.morphologyEx(amber_mask, cv2.MORPH_OPEN, kernel)

        amber_lights = self._extract_lights_from_mask(
            amber_mask, zone, zone_hsv, x1, y1, 'amber', timestamp
        )
        detected_lights.extend(amber_lights)

        # --- Interior light (warm white, larger area) ---
        # Detected as white but larger and more diffuse
        interior_candidates = [l for l in white_lights
                             if l.area > 500 and l.intensity < 0.8]
        for light in interior_candidates:
            light.light_type = VehicleLightType.INTERIOR

        # Store in history for persistence checking
        self._light_history.append({
            'lights': detected_lights,
            'timestamp': timestamp
        })

        # Filter for persistent lights (reduce false positives)
        persistent_lights = self._filter_persistent_lights(detected_lights)

        return persistent_lights

    def _extract_lights_from_mask(
        self,
        mask: np.ndarray,
        zone_bgr: np.ndarray,
        zone_hsv: np.ndarray,
        offset_x: int,
        offset_y: int,
        color: str,
        timestamp: float
    ) -> List[DetectedLight]:
        """Extract individual lights from a color mask."""
        lights = []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_light_area:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio filter - lights should be roughly circular or rectangular
            aspect = max(w, h) / (min(w, h) + 1)
            if aspect > 5:  # Too elongated
                continue

            # Get center and intensity
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Calculate intensity from brightness in the contour region
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)

            # Get mean value (brightness) in the contour
            mean_val = cv2.mean(zone_hsv[:, :, 2], mask=contour_mask)[0]
            intensity = mean_val / 255.0

            # Determine light type
            if color == 'red':
                # Brake lights are brighter than tail lights
                if intensity > 0.85:
                    light_type = VehicleLightType.BRAKE_LIGHT
                else:
                    light_type = VehicleLightType.TAIL_LIGHT
            elif color == 'white':
                # Check position - reverse lights at back, headlights at front
                # For now, classify based on position in zone
                zone_h = zone_bgr.shape[0]
                if cy > zone_h * 0.5:  # Lower half = back of car
                    light_type = VehicleLightType.REVERSE_LIGHT
                else:
                    light_type = VehicleLightType.HEADLIGHT
            elif color == 'amber':
                light_type = VehicleLightType.INDICATOR
            else:
                light_type = VehicleLightType.TAIL_LIGHT

            # Calculate confidence based on area, intensity, and shape
            size_score = min(1.0, area / 500)  # Larger = more confident
            shape_score = 1.0 - (aspect - 1) / 4  # More circular = better
            confidence = (intensity * 0.4 + size_score * 0.3 + shape_score * 0.3)

            lights.append(DetectedLight(
                light_type=light_type,
                location=(cx + offset_x, cy + offset_y),
                bbox=(x + offset_x, y + offset_y,
                      x + w + offset_x, y + h + offset_y),
                intensity=intensity,
                area=area,
                confidence=min(1.0, confidence),
                timestamp=timestamp
            ))

        return lights

    def _filter_persistent_lights(
        self,
        current_lights: List[DetectedLight]
    ) -> List[DetectedLight]:
        """
        Filter lights to those that persist across multiple frames.

        Reduces false positives from reflections or noise.
        """
        if len(self._light_history) < self.light_persistence_frames:
            return current_lights

        persistent = []

        for light in current_lights:
            # Count how many recent frames had a light near this location
            match_count = 0

            for hist_entry in list(self._light_history)[-self.light_persistence_frames:]:
                for hist_light in hist_entry['lights']:
                    # Check if same type and nearby location
                    if hist_light.light_type == light.light_type:
                        dist = np.sqrt(
                            (hist_light.location[0] - light.location[0]) ** 2 +
                            (hist_light.location[1] - light.location[1]) ** 2
                        )
                        if dist < 30:  # Within 30 pixels
                            match_count += 1
                            break

            # Light is persistent if seen in most recent frames
            if match_count >= self.light_persistence_frames - 1:
                light.confidence = min(1.0, light.confidence * 1.2)  # Boost confidence
                persistent.append(light)
            elif light.intensity > 0.9:
                # Very bright lights pass even without persistence (brake lights)
                persistent.append(light)

        return persistent

    def detect_paired_lights(
        self,
        lights: List[DetectedLight]
    ) -> List[Tuple[DetectedLight, DetectedLight]]:
        """
        Detect paired lights (e.g., two tail lights).

        Paired lights are a stronger indicator of a real vehicle.
        """
        pairs = []

        # Group by type
        by_type: Dict[VehicleLightType, List[DetectedLight]] = {}
        for light in lights:
            if light.light_type not in by_type:
                by_type[light.light_type] = []
            by_type[light.light_type].append(light)

        # Find pairs within each type
        for light_type, type_lights in by_type.items():
            if len(type_lights) < 2:
                continue

            # Sort by x position
            sorted_lights = sorted(type_lights, key=lambda l: l.location[0])

            for i, light1 in enumerate(sorted_lights):
                for light2 in sorted_lights[i+1:]:
                    # Check if horizontally aligned (similar y)
                    y_diff = abs(light1.location[1] - light2.location[1])
                    x_diff = abs(light1.location[0] - light2.location[0])

                    # Pairs should be horizontally separated but vertically aligned
                    if y_diff < 50 and 50 < x_diff < 400:
                        # Similar size
                        size_ratio = min(light1.area, light2.area) / max(light1.area, light2.area)
                        if size_ratio > 0.5:
                            pairs.append((light1, light2))

        return pairs

    def analyze_motion(
        self,
        frame: np.ndarray,
        car_zone: Tuple[int, int, int, int]
    ) -> MotionAnalysis:
        """
        Analyze motion in the car zone with direction detection.

        Uses:
        - Frame differencing for motion detection
        - Optical flow for motion direction
        - Size analysis for car-sized motion
        """
        timestamp = time.time()
        x1, y1, x2, y2 = car_zone

        # Validate bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return MotionAnalysis(
                motion_detected=False, motion_ratio=0.0, motion_center=None,
                motion_direction=None, flow_magnitude=0.0, is_significant=False,
                is_car_sized=False, timestamp=timestamp
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zone_gray = gray[y1:y2, x1:x2]

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            self._prev_frame = frame.copy()
            return MotionAnalysis(
                motion_detected=False, motion_ratio=0.0, motion_center=None,
                motion_direction=None, flow_magnitude=0.0, is_significant=False,
                is_car_sized=False, timestamp=timestamp
            )

        prev_zone_gray = self._prev_gray[y1:y2, x1:x2]

        # --- Frame Differencing ---
        # Blur to reduce noise
        zone_blur = cv2.GaussianBlur(zone_gray, (21, 21), 0)
        prev_blur = cv2.GaussianBlur(prev_zone_gray, (21, 21), 0)

        diff = cv2.absdiff(zone_blur, prev_blur)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Calculate motion metrics
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0

        motion_detected = motion_ratio > self.motion_threshold
        is_significant = motion_ratio > self.significant_motion_threshold

        # Find motion center
        motion_center = None
        M = cv2.moments(thresh)
        if M["m00"] > 0:
            motion_center = (
                int(M["m10"] / M["m00"]) + x1,
                int(M["m01"] / M["m00"]) + y1
            )

        # Estimate if motion is car-sized
        # Find largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        if contours:
            largest_area = max(cv2.contourArea(c) for c in contours)

        zone_area = (x2 - x1) * (y2 - y1)
        is_car_sized = largest_area > zone_area * 0.1  # >10% of zone

        # --- Optical Flow for Direction ---
        motion_direction = None
        flow_magnitude = 0.0

        if motion_detected:
            try:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_zone_gray, zone_gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )

                # Get average flow in motion region
                flow_mask = thresh > 0
                if np.any(flow_mask):
                    flow_x = flow[:, :, 0][flow_mask]
                    flow_y = flow[:, :, 1][flow_mask]

                    mean_flow_x = np.mean(flow_x)
                    mean_flow_y = np.mean(flow_y)

                    flow_magnitude = np.sqrt(mean_flow_x**2 + mean_flow_y**2)

                    # Determine direction
                    # Positive y = moving down = toward camera (car backing toward us)
                    # Negative y = moving up = away from camera (car driving away)
                    if flow_magnitude > 1.0:
                        if abs(mean_flow_y) > abs(mean_flow_x):
                            if mean_flow_y > 0:
                                motion_direction = 'toward_camera'
                            else:
                                motion_direction = 'away_from_camera'
                        else:
                            motion_direction = 'lateral'
                    else:
                        motion_direction = 'stationary'

            except Exception as e:
                logger.debug(f"Optical flow failed: {e}")

        # Update history
        self._prev_gray = gray.copy()
        self._prev_frame = frame.copy()

        result = MotionAnalysis(
            motion_detected=motion_detected,
            motion_ratio=motion_ratio,
            motion_center=motion_center,
            motion_direction=motion_direction,
            flow_magnitude=flow_magnitude,
            is_significant=is_significant,
            is_car_sized=is_car_sized,
            timestamp=timestamp
        )

        self._motion_history.append(result)

        return result

    def analyze_light_pattern(
        self,
        lights: List[DetectedLight]
    ) -> Optional[LightPattern]:
        """
        Analyze detected lights to determine vehicle state.

        Patterns:
        - parked: No lights or only parking lights
        - running: Tail lights on
        - reversing: Reverse lights on
        - departing: Tail lights + motion
        """
        if not lights:
            return None

        timestamp = time.time()

        # Count light types
        tail_count = sum(1 for l in lights if l.light_type == VehicleLightType.TAIL_LIGHT)
        brake_count = sum(1 for l in lights if l.light_type == VehicleLightType.BRAKE_LIGHT)
        reverse_count = sum(1 for l in lights if l.light_type == VehicleLightType.REVERSE_LIGHT)
        headlight_count = sum(1 for l in lights if l.light_type == VehicleLightType.HEADLIGHT)
        interior_count = sum(1 for l in lights if l.light_type == VehicleLightType.INTERIOR)

        # Determine pattern
        pattern_type = 'unknown'
        confidence = 0.5

        if reverse_count >= 1:
            pattern_type = 'reversing'
            confidence = 0.9 if reverse_count >= 2 else 0.75
        elif brake_count >= 2:
            pattern_type = 'braking'
            confidence = 0.85
        elif tail_count >= 2 or (tail_count >= 1 and (headlight_count >= 1 or brake_count >= 1)):
            pattern_type = 'running'
            confidence = 0.8 if tail_count >= 2 else 0.65
        elif interior_count >= 1:
            pattern_type = 'door_open'
            confidence = 0.7
        elif tail_count == 1:
            pattern_type = 'possibly_running'
            confidence = 0.5

        return LightPattern(
            pattern_type=pattern_type,
            lights=lights,
            confidence=confidence,
            timestamp=timestamp
        )

    def detect_departure_signals(
        self,
        frame: np.ndarray,
        car_zone: Tuple[int, int, int, int],
        car_detected_by_yolo: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point: Detect all departure signals in low light.

        Combines light detection, motion analysis, and pattern recognition.

        Returns dict with:
            - light_condition: Current lighting
            - lights: Detected vehicle lights
            - light_pairs: Paired lights found
            - light_pattern: Interpreted pattern
            - motion: Motion analysis results
            - departure_confidence: Overall confidence car is departing
            - signals: List of signal descriptions
        """
        # Detect light condition
        light_condition = self.detect_light_condition(frame)

        # Detect lights
        lights = self.detect_vehicle_lights(frame, car_zone)
        light_pairs = self.detect_paired_lights(lights)
        light_pattern = self.analyze_light_pattern(lights)

        # Analyze motion
        motion = self.analyze_motion(frame, car_zone)

        # Calculate departure confidence
        departure_confidence = 0.0
        signals = []

        # Light-based signals
        if light_pattern:
            if light_pattern.pattern_type == 'reversing':
                departure_confidence = max(departure_confidence, 0.9)
                signals.append(f"reverse_lights:{light_pattern.confidence:.2f}")
            elif light_pattern.pattern_type == 'running':
                departure_confidence = max(departure_confidence, 0.6)
                signals.append(f"running_lights:{light_pattern.confidence:.2f}")
            elif light_pattern.pattern_type == 'door_open':
                departure_confidence = max(departure_confidence, 0.4)
                signals.append(f"interior_light:{light_pattern.confidence:.2f}")

        # Paired lights boost confidence
        if len(light_pairs) >= 1:
            departure_confidence = min(1.0, departure_confidence * 1.2)
            signals.append(f"paired_lights:{len(light_pairs)}")

        # Motion signals
        if motion.is_significant and motion.is_car_sized:
            departure_confidence = max(departure_confidence, 0.7)
            signals.append(f"significant_motion:{motion.motion_ratio:.2f}")

            if motion.motion_direction:
                signals.append(f"direction:{motion.motion_direction}")
                # Backing out is strong departure signal
                if motion.motion_direction == 'toward_camera':
                    departure_confidence = min(1.0, departure_confidence + 0.2)
        elif motion.motion_detected:
            departure_confidence = max(departure_confidence, 0.4)
            signals.append(f"motion:{motion.motion_ratio:.2f}")

        # YOLO still seeing car reduces departure confidence
        if car_detected_by_yolo and light_condition in [LightCondition.DAYLIGHT, LightCondition.TWILIGHT]:
            # If we can still see the car, probably not departed yet
            if departure_confidence > 0.5:
                departure_confidence *= 0.7

        # Low light boosts signal confidence (signals more meaningful in dark)
        if light_condition in [LightCondition.LOW_LIGHT, LightCondition.NIGHT]:
            if lights:
                departure_confidence = min(1.0, departure_confidence * 1.1)

        return {
            'light_condition': light_condition,
            'lights': lights,
            'light_pairs': light_pairs,
            'light_pattern': light_pattern,
            'motion': motion,
            'departure_confidence': departure_confidence,
            'signals': signals
        }

    def get_status(self) -> Dict[str, Any]:
        """Get detector status for debugging."""
        return {
            'light_condition': self._current_light_condition.name,
            'brightness_history_len': len(self._brightness_history),
            'light_history_len': len(self._light_history),
            'motion_history_len': len(self._motion_history),
            'recent_lights': len(self._light_history[-1]['lights']) if self._light_history else 0
        }
