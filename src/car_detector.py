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
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import yaml

try:
    import cv2
except ImportError:
    cv2 = None

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

        # Load baseline position for position-based fallback
        self.baseline_bbox = None
        self.baseline_tolerance = 100  # pixels

        # Baseline image crops for model matching
        self._baseline_crops = []

        # False positive tracking
        self._false_positive_zones = []  # List of bbox regions that are NOT the car
        self._fp_data_path = Path(config.get("presence_tracking", {}).get(
            "baseline_path", "data/car_baseline.yaml"
        )).parent / "false_positives.yaml"

        self._load_baseline(config)
        self._load_baseline_images(config)
        self._load_false_positives()

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

    def _load_baseline(self, config: dict):
        """Load baseline car position from file for position-based detection."""
        baseline_path = config.get("presence_tracking", {}).get(
            "baseline_path", "data/car_baseline.yaml"
        )
        try:
            path = Path(baseline_path)
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f)
                if data and "baseline" in data:
                    bbox = data["baseline"].get("bbox", {})
                    if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                        self.baseline_bbox = (
                            bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        )
                        self.baseline_tolerance = data["baseline"].get(
                            "position_tolerance", 100
                        )
                        logger.info(f"Loaded baseline position: {self.baseline_bbox} "
                                   f"(tolerance: {self.baseline_tolerance}px)")
                    else:
                        logger.warning("Baseline file missing bbox coordinates")
            else:
                logger.info("No baseline file found - position fallback disabled")
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")

    def reload_baseline(self):
        """
        Reload baseline from disk.

        Call this when presence_tracker updates the baseline so that
        car_detector uses the current position for matching.
        """
        self._load_baseline(self.config)
        self._load_baseline_images(self.config)
        logger.info("Baseline reloaded from disk")

    def _load_baseline_images(self, config: dict):
        """Load baseline snapshot images and precompute features for model matching."""
        if cv2 is None:
            return

        baseline_dir = Path(config.get("presence_tracking", {}).get(
            "baseline_path", "data/car_baseline.yaml"
        )).parent / "baselines"

        if not baseline_dir.exists():
            logger.info("No baseline images directory found")
            return

        target_size = (128, 64)  # width, height - must match _match_model

        for img_path in sorted(baseline_dir.glob("baseline_*.jpg")):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Extract the car region from the baseline image using known bbox
                if self.baseline_bbox:
                    x1, y1, x2, y2 = self.baseline_bbox
                    # Clamp to image bounds
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        car_crop = img[y1:y2, x1:x2]
                    else:
                        continue
                else:
                    # Use center region as approximation
                    h, w = img.shape[:2]
                    car_crop = img[h // 4:3 * h // 4, w // 4:3 * w // 4]

                crop_resized = cv2.resize(car_crop, target_size)
                crop_hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
                crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

                h_hist = cv2.calcHist([crop_hsv], [0], None, [50], [0, 180])
                s_hist = cv2.calcHist([crop_hsv], [1], None, [60], [0, 256])
                v_hist = cv2.calcHist([crop_hsv], [2], None, [60], [0, 256])
                cv2.normalize(h_hist, h_hist)
                cv2.normalize(s_hist, s_hist)
                cv2.normalize(v_hist, v_hist)

                self._baseline_crops.append({
                    'path': str(img_path),
                    'h_hist': h_hist,
                    's_hist': s_hist,
                    'v_hist': v_hist,
                    'gray': crop_gray,
                })

            except Exception as e:
                logger.warning(f"Failed to load baseline image {img_path}: {e}")

        logger.info(f"Loaded {len(self._baseline_crops)} baseline image(s) for model matching")

    def _load_false_positives(self):
        """Load learned false positive zones."""
        try:
            if self._fp_data_path.exists():
                with open(self._fp_data_path) as f:
                    data = yaml.safe_load(f)
                if data and 'false_positive_zones' in data:
                    self._false_positive_zones = data['false_positive_zones']
                    logger.info(f"Loaded {len(self._false_positive_zones)} false positive zone(s)")
        except Exception as e:
            logger.warning(f"Failed to load false positives: {e}")

    def _save_false_positives(self):
        """Save learned false positive zones to disk."""
        try:
            self._fp_data_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'false_positive_zones': self._false_positive_zones,
                'count': len(self._false_positive_zones)
            }
            with open(self._fp_data_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            logger.debug(f"Saved {len(self._false_positive_zones)} false positive zone(s)")
        except Exception as e:
            logger.warning(f"Failed to save false positives: {e}")

    def record_false_positive(self, bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        Record a false positive detection. Called when user replies 'null'.

        Stores the bbox region so future detections in that area are penalised.
        Skips recording if the FP zone overlaps the car's baseline position
        (those zones provide no suppression value due to the baseline exception).
        """
        if bbox is None:
            bbox = self.car_bbox
        if bbox is None:
            logger.debug("No bbox to record as false positive")
            return

        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Don't record FP zones that overlap the baseline — they're ineffective
        # because _is_in_false_positive_zone skips detections at baseline anyway
        if self.baseline_bbox:
            bx = (self.baseline_bbox[0] + self.baseline_bbox[2]) // 2
            by = (self.baseline_bbox[1] + self.baseline_bbox[3]) // 2
            dist_to_baseline = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
            if dist_to_baseline < 50:
                logger.info(
                    f"Skipping FP zone at ({cx},{cy}) — overlaps baseline position "
                    f"(dist={dist_to_baseline:.0f}px). This FP is likely a vehicle "
                    f"contact issue, not a car detection issue."
                )
                return

        fp_entry = {
            'center': [cx, cy],
            'area': area,
            'bbox': list(bbox),
            'count': 1
        }

        # Check if this overlaps an existing FP zone (merge if so)
        merged = False
        for existing in self._false_positive_zones:
            ex_cx, ex_cy = existing['center']
            dist = ((cx - ex_cx) ** 2 + (cy - ex_cy) ** 2) ** 0.5
            if dist < 80:  # Within 80px = same zone
                existing['count'] = existing.get('count', 1) + 1
                # Update center as running average
                n = existing['count']
                existing['center'] = [
                    int((ex_cx * (n - 1) + cx) / n),
                    int((ex_cy * (n - 1) + cy) / n),
                ]
                merged = True
                logger.info(f"False positive zone updated (count={existing['count']}, "
                           f"center={existing['center']})")
                break

        if not merged:
            self._false_positive_zones.append(fp_entry)
            logger.info(f"New false positive zone recorded at ({cx}, {cy})")

        # Keep max 20 zones
        if len(self._false_positive_zones) > 20:
            self._false_positive_zones.sort(key=lambda z: z.get('count', 1))
            self._false_positive_zones = self._false_positive_zones[-20:]

        self._save_false_positives()

    def _is_in_false_positive_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if a detection bbox falls in a known false positive zone."""
        if not self._false_positive_zones:
            return False

        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        for fp_zone in self._false_positive_zones:
            fp_cx, fp_cy = fp_zone['center']
            dist = ((cx - fp_cx) ** 2 + (cy - fp_cy) ** 2) ** 0.5
            fp_count = fp_zone.get('count', 1)

            # More confirmed FPs = larger exclusion radius
            exclusion_radius = min(40 + fp_count * 10, 120)

            if dist < exclusion_radius:
                # Also check it's NOT at our baseline position
                if self.baseline_bbox:
                    bx = (self.baseline_bbox[0] + self.baseline_bbox[2]) // 2
                    by = (self.baseline_bbox[1] + self.baseline_bbox[3]) // 2
                    dist_to_baseline = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
                    if dist_to_baseline < 30:
                        # This is at our car's position, don't suppress
                        continue
                logger.debug(f"Detection at ({cx},{cy}) in FP zone "
                           f"(dist={dist:.0f}, count={fp_count})")
                return True

        return False

    def _matches_baseline_position(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Check if a detection matches the baseline position.

        Returns a confidence score (0.0 to 1.0) based on position match.
        """
        if self.baseline_bbox is None:
            return 0.0

        bx1, by1, bx2, by2 = bbox
        baseline_x1, baseline_y1, baseline_x2, baseline_y2 = self.baseline_bbox

        # Calculate centers
        det_cx = (bx1 + bx2) / 2
        det_cy = (by1 + by2) / 2
        base_cx = (baseline_x1 + baseline_x2) / 2
        base_cy = (baseline_y1 + baseline_y2) / 2

        # Distance between centers
        distance = ((det_cx - base_cx) ** 2 + (det_cy - base_cy) ** 2) ** 0.5

        # Calculate IoU for additional confidence
        inter_x1 = max(bx1, baseline_x1)
        inter_y1 = max(by1, baseline_y1)
        inter_x2 = min(bx2, baseline_x2)
        inter_y2 = min(by2, baseline_y2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (bx2 - bx1) * (by2 - by1)
            area2 = (baseline_x2 - baseline_x1) * (baseline_y2 - baseline_y1)
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
        else:
            iou = 0.0

        # Score based on distance and IoU
        # If distance is within tolerance, give high score
        if distance <= self.baseline_tolerance:
            distance_score = 1.0 - (distance / self.baseline_tolerance) * 0.3
        else:
            distance_score = max(0, 0.7 - (distance - self.baseline_tolerance) / 200)

        # Combine distance and IoU scores
        position_score = distance_score * 0.6 + iou * 0.4

        if position_score > 0.5:
            logger.debug(f"Position match: distance={distance:.0f}px, iou={iou:.2f}, "
                        f"score={position_score:.2f}")

        return position_score

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
        Run vehicle detection model using tiled detection.

        Uses overlapping tiles to detect smaller vehicles that would be
        missed when the full frame is resized to 640x640.

        Returns:
            List of (bbox, confidence) tuples for detected vehicles
        """
        model_name = "detector"
        if "car_detector" in self.hailo.models:
            model_name = "car_detector"

        frame_h, frame_w = frame.shape[:2]
        conf_threshold = self.config.get("detection", {}).get("proximity_confidence", 0.5)
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
        all_vehicles = []

        # Use tiled detection for wide frames (like 1920x1080)
        # This prevents small cars from being missed during resize
        if frame_w > 1200:
            # Tile 1: Left portion (0 to 60% width)
            tile1_end = int(frame_w * 0.6)
            tile1 = frame[:, 0:tile1_end]
            vehicles1 = self._detect_on_tile(tile1, model_name, conf_threshold, vehicle_classes, x_offset=0)
            all_vehicles.extend(vehicles1)

            # Tile 2: Right portion (40% to 100% width) - overlaps with tile 1
            tile2_start = int(frame_w * 0.4)
            tile2 = frame[:, tile2_start:]
            vehicles2 = self._detect_on_tile(tile2, model_name, conf_threshold, vehicle_classes, x_offset=tile2_start)
            all_vehicles.extend(vehicles2)

            # Remove duplicates via simple NMS
            all_vehicles = self._nms_vehicles(all_vehicles, iou_threshold=0.5)
        else:
            # Small frame - run on full frame
            all_vehicles = self._detect_on_tile(frame, model_name, conf_threshold, vehicle_classes, x_offset=0)

        return all_vehicles

    def _detect_on_tile(self, tile: np.ndarray, model_name: str, conf_threshold: float,
                        vehicle_classes: set, x_offset: int) -> List[Tuple[tuple, float]]:
        """Run detection on a single tile and adjust coordinates."""
        outputs = self.hailo.run_inference(model_name, tile)
        if outputs is None:
            return []

        tile_h, tile_w = tile.shape[:2]
        detections = postprocess_yolov8_detections(
            outputs,
            conf_threshold=conf_threshold,
            orig_shape=(tile_h, tile_w)
        )

        vehicles = []
        for det in detections:
            if det['class'] in vehicle_classes:
                x1, y1, x2, y2 = det['bbox']
                # Adjust x coordinates for tile offset
                adjusted_bbox = (x1 + x_offset, y1, x2 + x_offset, y2)
                vehicles.append((adjusted_bbox, det['confidence']))

        return vehicles

    def _nms_vehicles(self, vehicles: List[Tuple[tuple, float]], iou_threshold: float) -> List[Tuple[tuple, float]]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if not vehicles:
            return []

        # Sort by confidence (highest first)
        vehicles = sorted(vehicles, key=lambda x: x[1], reverse=True)
        keep = []

        while vehicles:
            best = vehicles.pop(0)
            keep.append(best)
            vehicles = [v for v in vehicles if self._iou(best[0], v[0]) < iou_threshold]

        return keep

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

        # 4. Position-based matching (uses baseline parking position)
        position_score = self._matches_baseline_position(bbox)
        if position_score >= 0.6:
            match_reasons.append(f"position_match:{position_score:.2f}")
        logger.debug(f"Position score: {position_score:.2f}")

        # Reject if in a known false positive zone (unless plate verified)
        if not plate_verified and self._is_in_false_positive_zone(bbox):
            logger.debug(f"Rejecting detection in false positive zone")
            return CarDetection(
                bbox=bbox,
                confidence=confidence,
                is_target_car=False,
                match_reasons=["rejected_fp_zone"],
                plate_verified=False
            )

        # Determine if this is the target car
        # Requires MULTIPLE corroborating signals, not just one weak match
        is_target = False
        final_confidence = confidence

        if plate_verified:
            # Plate is definitive
            is_target = True
            final_confidence = 1.0
        elif colour_score >= self.colour_threshold and model_score >= self.model_threshold:
            # Both colour AND model match = strong identification
            is_target = True
            final_confidence = (colour_score + model_score) / 2
        elif model_score >= 0.85 and position_score >= 0.7:
            # Strong model match at known position
            is_target = True
            final_confidence = (model_score * 0.6 + position_score * 0.4)
            match_reasons.append("model_position_match")
        elif colour_score >= self.colour_threshold and position_score >= 0.7:
            # Colour match at known position (requires tighter position)
            is_target = True
            final_confidence = (colour_score * 0.5 + position_score * 0.5) * 0.9
            match_reasons.append("colour_position_match")
        elif position_score >= 0.85 and colour_score >= 0.5:
            # Very tight position match with at least plausible colour
            # This handles cases where baseline is well-established
            is_target = True
            final_confidence = position_score * 0.9
            match_reasons.append("position_fallback")
            logger.info(f"Position fallback: vehicle at baseline position "
                       f"(pos={position_score:.2f}, colour={colour_score:.2f})")
        # Removed: lone colour match in zone is too weak
        # Removed: lone position >= 0.6 is too weak

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
                # Use higher threshold (70) to tolerate reflections from environment
                if detected_sat < 70:
                    if target_colour == 'white' and detected_val > 170:
                        return 0.9
                    elif target_colour in ['silver', 'grey', 'gray'] and 50 < detected_val < 200:
                        # Wider range (50-200) to handle varying lighting conditions
                        return 0.85
                    elif target_colour == 'black' and detected_val < 70:
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
        Match vehicle appearance against stored baseline images.

        Uses histogram comparison and structural similarity to determine
        if the detected vehicle looks like the owner's car. Compares against
        the baseline snapshots captured at different lighting conditions.

        Returns a confidence score (0.0 to 1.0).
        """
        if cv2 is None or vehicle_crop.size == 0:
            return 0.0

        if not self._baseline_crops:
            return 0.0

        try:
            # Resize crop to standard size for comparison
            target_size = (128, 64)  # width, height
            crop_resized = cv2.resize(vehicle_crop, target_size)
            crop_hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
            crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

            # Compute histograms for the candidate vehicle
            h_hist = cv2.calcHist([crop_hsv], [0], None, [50], [0, 180])
            s_hist = cv2.calcHist([crop_hsv], [1], None, [60], [0, 256])
            v_hist = cv2.calcHist([crop_hsv], [2], None, [60], [0, 256])
            cv2.normalize(h_hist, h_hist)
            cv2.normalize(s_hist, s_hist)
            cv2.normalize(v_hist, v_hist)

            best_score = 0.0

            for baseline_data in self._baseline_crops:
                b_h_hist = baseline_data['h_hist']
                b_s_hist = baseline_data['s_hist']
                b_v_hist = baseline_data['v_hist']
                b_gray = baseline_data['gray']

                # Histogram correlation (hue, saturation, value)
                h_corr = cv2.compareHist(h_hist, b_h_hist, cv2.HISTCMP_CORREL)
                s_corr = cv2.compareHist(s_hist, b_s_hist, cv2.HISTCMP_CORREL)
                v_corr = cv2.compareHist(v_hist, b_v_hist, cv2.HISTCMP_CORREL)

                # Structural similarity via normalized cross-correlation
                ncc = cv2.matchTemplate(crop_gray, b_gray, cv2.TM_CCORR_NORMED)
                struct_score = float(ncc[0][0]) if ncc.size > 0 else 0.0

                # Weight: hue matters most for car colour, structure for shape
                hist_score = (h_corr * 0.4 + s_corr * 0.2 + v_corr * 0.2)
                combined = hist_score * 0.5 + struct_score * 0.5

                # Clamp to 0-1
                combined = max(0.0, min(1.0, combined))
                best_score = max(best_score, combined)

            if best_score > 0.4:
                logger.debug(f"Model match score: {best_score:.2f}")

            return best_score

        except Exception as e:
            logger.warning(f"Model matching error: {e}")
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
