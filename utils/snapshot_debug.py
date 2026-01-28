#!/usr/bin/env python3
"""
Capture a debug snapshot showing car detection with bounding box.
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import yaml
from datetime import datetime

from camera import Camera
from car_detector import CarDetector
from utils.hailo_utils import HailoDevice


def main():
    # Load config
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    profile_path = PROJECT_ROOT / "config" / "car_profile.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(profile_path) as f:
        car_profile = yaml.safe_load(f)

    # Initialize Hailo
    print("Initializing Hailo...")
    hailo = HailoDevice()
    if not hailo.initialize():
        print("Failed to initialize Hailo")
        return 1

    # Load model
    model_path = PROJECT_ROOT / "models" / "yolov8s_h8l_nms.hef"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "yolov8n.hef"

    if model_path.exists():
        hailo.load_model(str(model_path), "detector")
        print(f"Loaded model: {model_path.name}")
    else:
        print("No model found!")
        return 1

    # Initialize detector
    print("Initializing car detector...")
    detector = CarDetector(config, car_profile, hailo)

    # Initialize camera
    print("Starting camera...")
    camera = Camera(
        resolution=(1920, 1080),
        framerate=15,
        use_picamera=True
    )

    if not camera.start():
        print("Failed to start camera")
        return 1

    # Capture frame
    print("Capturing frame...")
    frame = camera.capture()
    if frame is None:
        print("Failed to capture frame")
        camera.stop()
        return 1

    # Run detection
    print("Running detection...")
    detection = detector.detect(frame)

    # Draw results
    output = frame.copy()

    # Draw baseline position (yellow dashed)
    if detector.baseline_bbox:
        x1, y1, x2, y2 = detector.baseline_bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(output, "baseline", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw detection
    if detection and detection.is_target_car:
        x1, y1, x2, y2 = detection.bbox
        # Green box for target car
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Label with confidence
        label = f"YOUR CAR {detection.confidence:.0%}"
        cv2.putText(output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show match reasons
        reasons = ", ".join(detection.match_reasons)
        cv2.putText(output, reasons, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"TARGET CAR DETECTED!")
        print(f"  Confidence: {detection.confidence:.0%}")
        print(f"  Reasons: {detection.match_reasons}")
        print(f"  BBox: {detection.bbox}")
    else:
        print("Target car NOT detected in this frame")
        if detection:
            print(f"  (Detection exists but is_target_car=False)")
            print(f"  Reasons: {detection.match_reasons}")

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save
    output_path = PROJECT_ROOT / "data" / "debug" / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"\nSaved: {output_path}")

    # Cleanup
    camera.stop()
    hailo.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
