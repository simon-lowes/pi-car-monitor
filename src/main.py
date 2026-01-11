#!/usr/bin/env python3
"""
Pi Car Monitor - Main Entry Point
==================================
Privacy-respecting car security system that only records
when physical contact is made with the owner's vehicle.
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml

from pipeline import CarMonitorPipeline
from utils.hailo_utils import check_hailo_available


def setup_logging(config: dict) -> logging.Logger:
    """Configure logging based on config."""
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_file = log_config.get("file")
    
    # Create logs directory if needed
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger("car-monitor")


def load_config(config_path: str) -> dict:
    """Load main configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_car_profile(profile_path: str) -> dict:
    """Load car profile (sensitive data - local only)."""
    with open(profile_path, "r") as f:
        return yaml.safe_load(f)


def validate_setup(config: dict, car_profile: dict, logger: logging.Logger) -> bool:
    """Validate that all required components are ready."""
    
    # Check Hailo is available
    if config.get("performance", {}).get("use_hailo", True):
        if not check_hailo_available():
            logger.error("Hailo device not found. Run 'hailortcli fw-control identify' to diagnose.")
            return False
        logger.info("✓ Hailo device detected")
    
    # Check car profile is filled in
    vehicle = car_profile.get("vehicle", {})
    if not vehicle.get("make") or not vehicle.get("model"):
        logger.error("Car profile incomplete. Please fill in config/car_profile.yaml")
        return False
    logger.info(f"✓ Car profile loaded: {vehicle.get('year')} {vehicle.get('make')} {vehicle.get('model')}")
    
    # Check plate is configured
    plate = car_profile.get("plate", {})
    if not plate.get("number"):
        logger.warning("No plate number configured. Plate recognition will be disabled.")
    else:
        logger.info("✓ Plate recognition configured (stored as hash only)")
    
    # Check models exist
    models_dir = Path("models/custom")
    if not models_dir.exists() or not list(models_dir.glob("*.hef")):
        logger.warning("No custom models found. Run training/fine_tune.py first for best results.")
        logger.info("  Will use generic vehicle detection as fallback.")
    else:
        logger.info("✓ Custom detection models found")
    
    # Check recording directory
    rec_dir = Path(config.get("recording", {}).get("output_dir", "data/recordings"))
    rec_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Recording directory: {rec_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Pi Car Monitor - Privacy-First Car Security")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--profile", default="config/car_profile.yaml", help="Path to car profile")
    parser.add_argument("--test", action="store_true", help="Run in test mode (display output, don't record)")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration wizard")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without starting pipeline")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        car_profile = load_car_profile(args.profile)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Invalid YAML in configuration: {e}")
        sys.exit(1)
    
    # Enable debug mode for test runs
    if args.test:
        config["debug"]["enabled"] = True
        config["debug"]["display_output"] = True
        config["recording"]["enabled"] = False
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Pi Car Monitor starting...")
    
    # Validate setup
    if not validate_setup(config, car_profile, logger):
        logger.error("Setup validation failed. Please fix the issues above.")
        sys.exit(1)
    
    if args.dry_run:
        logger.info("Dry run complete. Setup looks good!")
        sys.exit(0)
    
    if args.calibrate:
        from training.calibrate import run_calibration_wizard
        run_calibration_wizard(config, car_profile)
        sys.exit(0)
    
    # Create and start pipeline
    pipeline = CarMonitorPipeline(config, car_profile)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    try:
        logger.info("Starting car monitoring pipeline...")
        logger.info("Press Ctrl+C to stop")
        pipeline.start()
        pipeline.wait()  # Block until pipeline stops
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        pipeline.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
