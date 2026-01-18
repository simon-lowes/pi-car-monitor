"""
Car Detector - Stage 1
======================
Identifies whether the owner's specific car is in frame.

Uses a combination of:
1. General vehicle detection (YOLOv8)
2. Colour matching
3. Shape/model matching (fine-tuned model if available)
4. Number plate verification (definitive, local-only)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

from plate_handler import PlateHandler
from utils.hailo_utils import postprocess_yolov8_detections

logger = logging.getLogger(__name__)


@dataclass
class CarDetection:
    """Represents a detected vehicle."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    is_target_car: bool
    match_reasons: list[str]
    plate_verified: bool = False


class CarDetector:
    """
    Detects and identifies the owner's specific car.
    
    This is Stage 1 of the detection pipeline. It runs on every frame
    (or every Nth frame for efficiency) and determines if the owner's
    car is visible in the camera feed.
    """
    
    def __init__(self, config: dict, car_profile: dict, hailo_device):
        self.config = config
        self.car_profile = car_profile
        self.hailo = hailo_device
        
        # Extract car details
        vehicle = car_profile.get("vehicle", {})
        self.target_make = vehicle.get("make", "").lower()
        self.target_model = vehicle.get("model", "").lower()
        self.target_year = vehicle.get("year")
        self.target_colour = vehicle.get("colour", {}).get("primary", "").lower()
        self.target_body_style = vehicle.get("body_style", "").lower()
        
        # Recognition thresholds
        recognition = car_profile.get("recognition", {})
        self.model_threshold = recognition.get("model_match_threshold", 0.7)
        self.colour_threshold = recognition.get("colour_match_threshold", 0.8)
        self.plate_threshold = recognition.get("plate_match_threshold", 0.95)
        self.plate_overrides = recognition.get("plate_overrides_visual", True)
        
        # Initialize plate handler (local-only, secure)
        plate_config = car_profile.get("plate", {})
        if plate_config.get("number"):
            self.plate_handler = PlateHandler(
                target_plate=plate_config["number"],
                salt_file=config.get("plate_recognition", {}).get("salt_file")
            )
            logger.info("Plate verification enabled (hash-based, local only)")
        else:
            self.plate_handler = None
            logger.info("Plate verification disabled - no plate configured")
        
        # Detection zone (will be set during calibration)
        zones = config.get("zones", {})
        car_zone = zones.get("car_zone", {})
        self.zone_enabled = car_zone.get("enabled", True)
        self.zone_bounds = (
            car_zone.get("x_min", 0.0),
            car_zone.get("y_min", 0.0),
            car_zone.get("x_max", 1.0),
            car_zone.get("y_max", 1.0)
        )
        
        # State tracking
        self.last_detection: Optional[CarDetection] = None
        self.consecutive_detections = 0
        self.detection_stable = False
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load detection models onto Hailo."""
        # TODO: Load the actual HEF models
        # For now, this is a placeholder structure
        
        # Try custom fine-tuned model first
        custom_model_path = "models/custom/car_detector.hef"
        generic_model_path = "models/yolov8n.hef"
        
        # The actual loading will be:
        # self.detector_model = self.hailo.load_model(model_path)
        
        logger.info("Detection models loaded")
    
    def detect(self, frame: np.ndarray) -> Optional[CarDetection]:
        """
        Process a frame and detect if the target car is present.

        Args:
            frame: BGR image as numpy array

        Returns:
            CarDetection if target car found, None otherwise
        """
        frame_h, frame_w = frame.shape[:2]

        # Run vehicle detection
        vehicles = self._detect_vehicles(frame)

        if not vehicles:
            self._update_state(None)
            return None

        logger.debug(f"Detected {len(vehicles)} vehicle(s) in frame")
        
        # Filter to detection zone if enabled
        if self.zone_enabled:
            vehicles = self._filter_to_zone(vehicles, frame_w, frame_h)
        
        # Check each detected vehicle
        for vehicle_bbox, vehicle_conf in vehicles:
            detection = self._check_if_target(frame, vehicle_bbox, vehicle_conf)
            if detection and detection.is_target_car:
                self._update_state(detection)
                return detection
        
        self._update_state(None)
        return None
    
    def _detect_vehicles(self, frame: np.ndarray) -> List[Tuple[tuple, float]]:
        """
        Run vehicle detection model.

        Returns:
            List of (bbox, confidence) tuples for detected vehicles
        """
        # Run inference
        model_name = "detector"
        if "car_detector" in self.hailo.models:
            model_name = "car_detector"

        outputs = self.hailo.run_inference(model_name, frame)

        if outputs is None:
            logger.warning("Hailo inference returned None")
            return []

        # Log output structure for debugging
        for key, val in outputs.items():
            if isinstance(val, list):
                logger.debug(f"Output '{key}': list with {len(val)} items")
                if val and isinstance(val[0], list):
                    logger.debug(f"  First item has {len(val[0])} class arrays")
            elif hasattr(val, 'shape'):
                logger.debug(f"Output '{key}': array shape {val.shape}")

        # Postprocess detections
        frame_h, frame_w = frame.shape[:2]
        detections = postprocess_yolov8_detections(
            outputs,
            conf_threshold=self.config.get("detection", {}).get("proximity_confidence", 0.5),
            orig_shape=(frame_h, frame_w)
        )

        # Log all detections before filtering
        if detections:
            det_summary = [(d['class'], round(d['confidence'], 2)) for d in detections]
            logger.debug(f"Raw detections: {det_summary}")

        # Filter to vehicle classes only
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
        vehicles = []

        for det in detections:
            if det['class'] in vehicle_classes:
                vehicles.append((det['bbox'], det['confidence']))

        return vehicles
    
    def _filter_to_zone(self, vehicles: list, frame_w: int, frame_h: int) -> list:
        """Filter detections to only those within the configured zone."""
        x_min = int(self.zone_bounds[0] * frame_w)
        y_min = int(self.zone_bounds[1] * frame_h)
        x_max = int(self.zone_bounds[2] * frame_w)
        y_max = int(self.zone_bounds[3] * frame_h)
        
        filtered = []
        for bbox, conf in vehicles:
            bx1, by1, bx2, by2 = bbox
            # Check if center of detection is in zone
            center_x = (bx1 + bx2) // 2
            center_y = (by1 + by2) // 2
            
            if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                filtered.append((bbox, conf))
        
        return filtered
    
    def _check_if_target(self, frame: np.ndarray, bbox: tuple, confidence: float) -> CarDetection:
        """
        Determine if a detected vehicle is the owner's target car.
        
        Uses multiple matching strategies:
        1. Plate recognition (definitive if readable)
        2. Colour matching
        3. Model/shape matching
        """
        x1, y1, x2, y2 = bbox
        vehicle_crop = frame[y1:y2, x1:x2]
        
        match_reasons = []
        plate_verified = False
        
        # 1. Try plate recognition first (most reliable)
        if self.plate_handler:
            plate_result = self.plate_handler.check_plate(vehicle_crop)
            if plate_result.is_match:
                plate_verified = True
                match_reasons.append("plate_match")
                logger.debug("Target car confirmed via plate match")
                
                if self.plate_overrides:
                    # Plate match is definitive
                    return CarDetection(
                        bbox=bbox,
                        confidence=1.0,
                        is_target_car=True,
                        match_reasons=match_reasons,
                        plate_verified=True
                    )
        
        # 2. Colour matching
        colour_score = self._match_colour(vehicle_crop)
        if colour_score >= self.colour_threshold:
            match_reasons.append(f"colour_match:{colour_score:.2f}")
        logger.debug(f"Vehicle colour score: {colour_score:.2f} (threshold: {self.colour_threshold})")

        # 3. Model/shape matching (if custom model available)
        model_score = self._match_model(vehicle_crop)
        if model_score >= self.model_threshold:
            match_reasons.append(f"model_match:{model_score:.2f}")
        logger.debug(f"Model score: {model_score:.2f} (threshold: {self.model_threshold})")
        
        # Determine if this is the target car
        is_target = False
        final_confidence = confidence

        if plate_verified:
            is_target = True
            final_confidence = 1.0
        elif colour_score >= self.colour_threshold and model_score >= self.model_threshold:
            is_target = True
            final_confidence = (colour_score + model_score) / 2
        elif colour_score >= 0.9 or model_score >= 0.9:
            # Very high single-factor match
            is_target = True
            final_confidence = max(colour_score, model_score)
        elif colour_score >= self.colour_threshold and self.zone_enabled:
            # Fallback: colour match + in designated zone = likely target car
            # This allows detection when plate/model matching unavailable
            is_target = True
            final_confidence = colour_score * 0.9  # Slightly lower confidence
            match_reasons.append("zone_fallback")
            logger.debug(f"Zone fallback triggered: colour {colour_score:.2f} in zone")

        if is_target:
            logger.info(f"Target car detected! Confidence: {final_confidence:.2f}, reasons: {match_reasons}")

        return CarDetection(
            bbox=bbox,
            confidence=final_confidence,
            is_target_car=is_target,
            match_reasons=match_reasons,
            plate_verified=plate_verified
        )
    
    def _match_colour(self, vehicle_crop: np.ndarray) -> float:
        """
        Match the vehicle's colour against target.

        Uses HSV colour space for robustness to lighting changes.
        """
        try:
            import cv2

            if vehicle_crop.size == 0:
                return 0.0

            # Convert to HSV
            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)

            # Get dominant colour by histogram analysis
            # Focus on the center portion (avoid edges/shadows)
            h, w = hsv.shape[:2]
            center_region = hsv[h//4:3*h//4, w//4:3*w//4]

            if center_region.size == 0:
                return 0.0

            # Calculate mean hue and saturation
            mean_hsv = np.mean(center_region, axis=(0, 1))
            detected_hue = mean_hsv[0]
            detected_sat = mean_hsv[1]
            detected_val = mean_hsv[2]

            # Define colour ranges (HSV values for common car colours)
            # Hue is 0-180 in OpenCV
            COLOUR_RANGES = {
                'red': [(0, 10), (160, 180)],  # Red wraps around
                'orange': [(10, 25)],
                'yellow': [(25, 35)],
                'green': [(35, 85)],
                'blue': [(85, 130)],
                'purple': [(130, 160)],
                'white': None,   # High value, low saturation
                'black': None,   # Low value
                'silver': None,  # Medium value, low saturation
                'grey': None,    # Medium value, low saturation
                'gray': None,    # Alias
            }

            target_colour = self.target_colour.lower()

            # Handle achromatic colours (white, black, silver, grey)
            if target_colour in ['white', 'silver', 'grey', 'gray']:
                # Low saturation indicates achromatic
                # Use higher threshold (60) to tolerate reflections from environment
                if detected_sat < 60:
                    if target_colour == 'white' and detected_val > 180:
                        return 0.9
                    elif target_colour in ['silver', 'grey', 'gray'] and 80 < detected_val < 180:
                        return 0.85
                    elif target_colour == 'black' and detected_val < 60:
                        return 0.9
                return 0.3

            if target_colour == 'black':
                if detected_val < 60:
                    return 0.9
                return 0.2

            # Check chromatic colours
            if target_colour in COLOUR_RANGES and COLOUR_RANGES[target_colour]:
                ranges = COLOUR_RANGES[target_colour]
                for hue_min, hue_max in ranges:
                    if hue_min <= detected_hue <= hue_max:
                        # Saturation should be reasonably high for chromatic match
                        sat_factor = min(detected_sat / 100, 1.0)
                        return 0.7 + 0.3 * sat_factor

            return 0.3  # Low confidence for unknown/non-matching colour

        except Exception as e:
            logger.warning(f"Colour matching error: {e}")
            return 0.0
    
    def _match_model(self, vehicle_crop: np.ndarray) -> float:
        """
        Match vehicle shape/model using fine-tuned classifier.
        
        If no custom model is available, returns 0.0 (not used).
        """
        # TODO: Run model classifier if available
        # This would use a model trained on images of the specific
        # make/model to identify it
        
        # Placeholder
        return 0.0
    
    def _update_state(self, detection: Optional[CarDetection]):
        """Update detection state for stability tracking."""
        if detection and detection.is_target_car:
            self.consecutive_detections += 1
            self.last_detection = detection
            
            # Require multiple consecutive detections for stability
            if self.consecutive_detections >= 3:
                self.detection_stable = True
        else:
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            if self.consecutive_detections == 0:
                self.detection_stable = False
                self.last_detection = None
    
    @property
    def car_in_frame(self) -> bool:
        """Returns True if target car is stably detected."""
        return self.detection_stable
    
    @property
    def car_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns bounding box of target car if detected."""
        if self.last_detection:
            return self.last_detection.bbox
        return None
