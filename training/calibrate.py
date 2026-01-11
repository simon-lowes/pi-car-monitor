#!/usr/bin/env python3
"""
Calibration Wizard
==================
Interactive calibration for setting up the car monitoring system.

Steps:
1. Camera check - verify camera is working
2. Car positioning - capture images of your car from the camera angle
3. Zone definition - define the detection zone for your parking space
4. Proximity calibration - set the proximity buffer distance
5. Save configuration

Run with: python training/calibrate.py
Or via: python src/main.py --calibrate
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

# Add project paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)


def run_calibration_wizard(config: dict, car_profile: dict):
    """Run the interactive calibration wizard."""
    print("\n" + "=" * 50)
    print("  Pi Car Monitor - Calibration Wizard")
    print("=" * 50 + "\n")

    calibration_dir = Path("data/calibration")
    calibration_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Camera Check
    print("Step 1/5: Camera Check")
    print("-" * 30)
    camera = check_camera(config)
    if camera is None:
        print("ERROR: Camera not available. Please check your camera connection.")
        return False

    print("Camera working.\n")

    # Step 2: Car Positioning
    print("Step 2/5: Car Positioning")
    print("-" * 30)
    print("Please ensure your car is parked in its usual position.")
    print("We'll capture calibration images from this angle.\n")

    input("Press Enter when your car is positioned correctly...")

    # Capture calibration images
    calibration_images = capture_calibration_images(camera, calibration_dir)
    print(f"Captured {len(calibration_images)} calibration images.\n")

    # Step 3: Zone Definition
    print("Step 3/5: Detection Zone Definition")
    print("-" * 30)
    print("Define the region where your car is typically parked.")

    zone = define_detection_zone(camera, config)
    print(f"Zone defined: {zone}\n")

    # Step 4: Proximity Buffer
    print("Step 4/5: Proximity Buffer")
    print("-" * 30)
    print("Set how close someone must be to your car before tracking starts.")
    print("This is measured in pixels from the car's bounding box edge.")

    proximity_buffer = set_proximity_buffer(config)
    print(f"Proximity buffer: {proximity_buffer} pixels\n")

    # Step 5: Save Configuration
    print("Step 5/5: Save Configuration")
    print("-" * 30)

    # Update config
    updated_config = update_config_with_calibration(
        config,
        zone,
        proximity_buffer
    )

    # Save calibration data
    save_calibration_data(
        calibration_dir,
        zone,
        proximity_buffer,
        calibration_images
    )

    print("\nCalibration complete!")
    print(f"Calibration images saved to: {calibration_dir}")
    print("\nNext steps:")
    print("1. Review the calibration images in data/calibration/")
    print("2. Run: python training/fine_tune.py (optional, for better accuracy)")
    print("3. Start monitoring: python src/main.py")

    # Cleanup
    camera.stop()

    return True


def check_camera(config: dict):
    """Check if camera is working and return camera instance."""
    try:
        from camera import Camera

        cam_config = config.get("camera", {})
        resolution = cam_config.get("resolution", {})

        camera = Camera(
            device=cam_config.get("device", "/dev/video0"),
            resolution=(
                resolution.get("width", 1920),
                resolution.get("height", 1080)
            ),
            framerate=cam_config.get("framerate", 15)
        )

        if not camera.start():
            return None

        # Test capture
        frame = camera.capture()
        if frame is None:
            camera.stop()
            return None

        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"  Backend: {'picamera2' if camera.use_picamera else 'opencv'}")

        return camera

    except Exception as e:
        print(f"Camera error: {e}")
        return None


def capture_calibration_images(camera, output_dir: Path) -> list:
    """Capture calibration images from different times/conditions."""
    images = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nCapturing calibration images...")
    print("(For best results, also capture images at different times of day)")

    for i in range(5):
        frame = camera.capture()
        if frame is not None:
            filename = f"calibration_{timestamp}_{i+1}.jpg"
            filepath = output_dir / filename

            try:
                import cv2
                cv2.imwrite(str(filepath), frame)
                images.append(str(filepath))
                print(f"  Captured: {filename}")
            except Exception as e:
                print(f"  Failed to save image: {e}")

        time.sleep(0.5)

    return images


def define_detection_zone(camera, config: dict) -> dict:
    """
    Define the detection zone interactively.

    Returns zone as percentage coordinates.
    """
    print("\nDefining detection zone...")
    print("The zone should cover where your car is parked.\n")

    # Get current zone from config
    current_zone = config.get("zones", {}).get("car_zone", {})

    # Default zone
    zone = {
        "enabled": True,
        "x_min": current_zone.get("x_min", 0.2),
        "x_max": current_zone.get("x_max", 0.8),
        "y_min": current_zone.get("y_min", 0.3),
        "y_max": current_zone.get("y_max", 0.9)
    }

    print("Current zone (as percentage of frame):")
    print(f"  X: {zone['x_min']*100:.0f}% - {zone['x_max']*100:.0f}%")
    print(f"  Y: {zone['y_min']*100:.0f}% - {zone['y_max']*100:.0f}%")

    # Option to use visual calibration
    response = input("\nUse visual zone editor? (requires display) [y/N]: ").lower()

    if response == 'y':
        zone = visual_zone_editor(camera, zone)
    else:
        # Manual entry
        response = input("Adjust zone manually? [y/N]: ").lower()
        if response == 'y':
            zone = manual_zone_entry(zone)

    return zone


def visual_zone_editor(camera, current_zone: dict) -> dict:
    """Visual zone editor using OpenCV window."""
    try:
        import cv2

        # Capture a reference frame
        frame = camera.capture()
        if frame is None:
            print("Failed to capture frame. Using manual entry.")
            return manual_zone_entry(current_zone)

        h, w = frame.shape[:2]

        # Zone in pixels
        zone_px = {
            "x1": int(current_zone["x_min"] * w),
            "y1": int(current_zone["y_min"] * h),
            "x2": int(current_zone["x_max"] * w),
            "y2": int(current_zone["y_max"] * h)
        }

        # Mouse callback state
        state = {
            "drawing": False,
            "start": None,
            "zone": zone_px.copy()
        }

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drawing"] = True
                state["start"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
                state["zone"]["x1"] = min(state["start"][0], x)
                state["zone"]["y1"] = min(state["start"][1], y)
                state["zone"]["x2"] = max(state["start"][0], x)
                state["zone"]["y2"] = max(state["start"][1], y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["drawing"] = False

        cv2.namedWindow("Zone Calibration")
        cv2.setMouseCallback("Zone Calibration", mouse_callback)

        print("\nZone Editor:")
        print("  - Click and drag to draw the detection zone")
        print("  - Press 's' to save and exit")
        print("  - Press 'r' to reset")
        print("  - Press 'q' to cancel")

        while True:
            # Get fresh frame
            display_frame = camera.capture()
            if display_frame is None:
                display_frame = frame.copy()
            else:
                display_frame = display_frame.copy()

            # Draw zone
            z = state["zone"]
            cv2.rectangle(
                display_frame,
                (z["x1"], z["y1"]),
                (z["x2"], z["y2"]),
                (0, 255, 0),
                2
            )

            # Add instructions
            cv2.putText(
                display_frame,
                "Draw zone, then press 's' to save",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("Zone Calibration", display_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('r'):
                state["zone"] = zone_px.copy()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return current_zone

        cv2.destroyAllWindows()

        # Convert back to percentages
        return {
            "enabled": True,
            "x_min": state["zone"]["x1"] / w,
            "x_max": state["zone"]["x2"] / w,
            "y_min": state["zone"]["y1"] / h,
            "y_max": state["zone"]["y2"] / h
        }

    except Exception as e:
        print(f"Visual editor error: {e}")
        print("Falling back to manual entry.")
        return manual_zone_entry(current_zone)


def manual_zone_entry(current_zone: dict) -> dict:
    """Manual zone entry via text prompts."""
    print("\nEnter zone as percentage of frame (0-100):")

    try:
        x_min = float(input(f"  X minimum [{current_zone['x_min']*100:.0f}]: ") or current_zone['x_min']*100) / 100
        x_max = float(input(f"  X maximum [{current_zone['x_max']*100:.0f}]: ") or current_zone['x_max']*100) / 100
        y_min = float(input(f"  Y minimum [{current_zone['y_min']*100:.0f}]: ") or current_zone['y_min']*100) / 100
        y_max = float(input(f"  Y maximum [{current_zone['y_max']*100:.0f}]: ") or current_zone['y_max']*100) / 100

        return {
            "enabled": True,
            "x_min": max(0, min(1, x_min)),
            "x_max": max(0, min(1, x_max)),
            "y_min": max(0, min(1, y_min)),
            "y_max": max(0, min(1, y_max))
        }
    except ValueError:
        print("Invalid input. Using current values.")
        return current_zone


def set_proximity_buffer(config: dict) -> int:
    """Set the proximity buffer distance."""
    current = config.get("zones", {}).get("proximity_buffer", 50)

    print(f"\nCurrent proximity buffer: {current} pixels")
    print("Typical values:")
    print("  30-50 pixels = close proximity (about 0.5m at typical camera distance)")
    print("  50-100 pixels = medium proximity (about 1m)")
    print("  100-150 pixels = far proximity (about 1.5-2m)")

    try:
        new_value = input(f"\nNew proximity buffer [{current}]: ")
        if new_value:
            return int(new_value)
    except ValueError:
        pass

    return current


def update_config_with_calibration(config: dict, zone: dict, proximity_buffer: int) -> dict:
    """Update config with calibration values."""
    config_path = Path("config/config.yaml")

    if config_path.exists():
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
    else:
        full_config = config.copy()

    # Update zone
    if "zones" not in full_config:
        full_config["zones"] = {}

    full_config["zones"]["car_zone"] = zone
    full_config["zones"]["proximity_buffer"] = proximity_buffer

    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(full_config, f, default_flow_style=False)

    print(f"Configuration saved to {config_path}")

    return full_config


def save_calibration_data(
    output_dir: Path,
    zone: dict,
    proximity_buffer: int,
    calibration_images: list
):
    """Save calibration metadata."""
    metadata = {
        "calibration_date": datetime.now().isoformat(),
        "zone": zone,
        "proximity_buffer": proximity_buffer,
        "calibration_images": calibration_images,
    }

    metadata_path = output_dir / "calibration_metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    print(f"Calibration metadata saved to {metadata_path}")


def main():
    """Standalone calibration runner."""
    parser = argparse.ArgumentParser(description="Pi Car Monitor Calibration")
    parser.add_argument("--config", default="config/config.yaml", help="Config file")
    parser.add_argument("--profile", default="config/car_profile.yaml", help="Car profile")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Load config
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        with open(args.profile, "r") as f:
            car_profile = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Config file not found: {e}")
        sys.exit(1)

    # Run calibration
    success = run_calibration_wizard(config, car_profile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
