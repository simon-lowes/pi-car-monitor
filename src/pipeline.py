"""
Car Monitor Pipeline
====================
Orchestrates the multi-stage detection pipeline:
1. Car Detection → 2. Proximity Tracking → 3. Contact Classification → 4. Recording

Uses GStreamer with Hailo plugins for efficient video processing.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import queue

import cv2

from camera import ThreadedCamera
from car_detector import CarDetector
from proximity_tracker import ProximityTracker
from contact_classifier import ContactClassifier, ContactType
from recorder import CircularRecorder
from database import EventDatabase
from utils.hailo_utils import HailoDevice
from telegram_notifier import TelegramNotifier, create_notifier_from_config
from owner_profile import OwnerProfile, create_profile_from_config

logger = logging.getLogger(__name__)


class CarMonitorPipeline:
    """
    Main pipeline that coordinates all detection and recording.
    
    This runs continuously, processing frames from the camera and
    triggering recordings only when contact with the target car is detected.
    """
    
    def __init__(self, config: dict, car_profile: dict):
        self.config = config
        self.car_profile = car_profile

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Frame queue for processing
        self._frame_queue = queue.Queue(maxsize=10)

        # Initialize camera
        cam_config = config.get("camera", {})
        resolution = cam_config.get("resolution", {})
        self.camera = ThreadedCamera(
            device=cam_config.get("device", "/dev/video0"),
            resolution=(
                resolution.get("width", 1920),
                resolution.get("height", 1080)
            ),
            framerate=cam_config.get("framerate", 15)
        )

        # Initialize Hailo device
        self.hailo = HailoDevice(
            config.get("performance", {}).get("hailo_device", "/dev/hailo0")
        )
        
        # Initialize detection stages
        self.car_detector = CarDetector(config, car_profile, self.hailo)
        self.proximity_tracker = ProximityTracker(config, self.hailo)
        self.contact_classifier = ContactClassifier(
            config,
            car_bbox_getter=lambda: self.car_detector.car_bbox
        )
        
        # Initialize recorder
        rec_config = config.get("recording", {})
        self.recorder = CircularRecorder(
            output_dir=rec_config.get("output_dir", "data/recordings"),
            pre_buffer_seconds=rec_config.get("pre_buffer_seconds", 10),
            post_buffer_seconds=rec_config.get("post_buffer_seconds", 5),
            max_duration=rec_config.get("max_duration_seconds", 300),
            format=rec_config.get("format", "mp4")
        )

        # Initialize database
        db_path = config.get("logging", {}).get(
            "events_db",
            "logs/events.db"
        )
        self.database = EventDatabase(db_path)
        self._session_id: Optional[int] = None

        # Initialize Telegram notifier
        self.notifier = create_notifier_from_config(config)
        if self.notifier:
            logger.info("Telegram notifications enabled")

        # Initialize owner profile for recognition
        self.owner_profile = create_profile_from_config(config)
        if self.owner_profile:
            logger.info(f"Owner recognition enabled ({self.owner_profile.sample_count} samples)")

        # Store last frame for snapshot
        self._last_frame = None
        
        # Performance settings
        perf_config = config.get("performance", {})
        self.detection_frame_skip = perf_config.get("detection_frame_skip", 2)
        self.proximity_frame_skip = perf_config.get("proximity_frame_skip", 1)
        
        # State tracking
        self.frame_count = 0
        self.last_contact_time = 0
        self.current_state = "initializing"
        
        # Statistics
        self.stats = {
            "frames_processed": 0,
            "car_detections": 0,
            "proximity_events": 0,
            "contact_events": 0,
            "recordings_triggered": 0,
            "start_time": None,
        }
    
    def start(self):
        """Start the monitoring pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return

        # Start camera
        if not self.camera.start():
            logger.error("Failed to start camera")
            raise RuntimeError("Camera initialization failed")
        logger.info("Camera started")

        # Initialize Hailo
        if self.config.get("performance", {}).get("use_hailo", True):
            if not self.hailo.initialize():
                logger.error("Failed to initialize Hailo device")
                raise RuntimeError("Hailo initialization failed")

            # Load models
            self._load_models()

        # Start recorder (maintains circular buffer)
        self.recorder.start()
        
        # Start processing thread
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        self.stats["start_time"] = datetime.now()
        self.current_state = "monitoring"

        # Start database session
        self._session_id = self.database.start_session()

        # Start Telegram reply listener for owner recognition
        if self.notifier and self.owner_profile:
            self.notifier.start_listener(self._on_owner_confirmed)

        logger.info("Pipeline started")
    
    def stop(self):
        """Stop the monitoring pipeline."""
        if not self._running:
            return

        logger.info("Stopping pipeline...")
        self._running = False
        self._stop_event.set()

        # Stop Telegram listener
        if self.notifier:
            self.notifier.stop_listener()

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Stop camera
        self.camera.stop()

        # Stop recorder
        self.recorder.stop()

        # Cleanup Hailo
        self.hailo.cleanup()

        # End database session
        if self._session_id:
            self.database.end_session(self._session_id, self.stats)
            self._session_id = None

        self.current_state = "stopped"
        logger.info("Pipeline stopped")
    
    def wait(self):
        """Block until pipeline stops."""
        if self._thread:
            self._thread.join()
    
    def _load_models(self):
        """Load detection models onto Hailo."""
        models_dir = Path("models")
        
        # Try custom models first
        custom_car_model = models_dir / "custom" / "car_detector.hef"
        if custom_car_model.exists():
            self.hailo.load_model(str(custom_car_model), "car_detector")
        else:
            # Fall back to generic YOLOv8
            generic_model = models_dir / "yolov8n.hef"
            if generic_model.exists():
                self.hailo.load_model(str(generic_model), "detector")
            else:
                logger.warning("No detection models found")
        
        # Pose estimation for contact detection
        pose_model = models_dir / "yolov8n_pose.hef"
        if pose_model.exists():
            self.hailo.load_model(str(pose_model), "pose")
        
        logger.info("Models loaded")
    
    def _run_loop(self):
        """Main processing loop."""
        # TODO: Set up GStreamer pipeline for camera capture
        # For now, this is a simplified loop structure
        
        logger.info("Processing loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get frame from camera
                frame = self._capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_count += 1
                timestamp = time.time()
                
                # Add frame to recorder buffer
                self.recorder.add_frame(frame, timestamp)

                # Store frame for notification snapshots
                self._last_frame = frame
                
                # Determine frame skip based on current state
                if self.car_detector.car_in_frame:
                    skip = self.proximity_frame_skip
                else:
                    skip = self.detection_frame_skip
                
                # Skip frames for efficiency
                if self.frame_count % skip != 0:
                    continue
                
                # Stage 1: Car Detection
                car_detection = self.car_detector.detect(frame)

                # Log status every 150 frames (~10 seconds at 15fps)
                if self.frame_count % 150 == 0:
                    logger.info(f"Status: state={self.current_state}, frames={self.frame_count}, "
                               f"car_in_frame={self.car_detector.car_in_frame}, "
                               f"detections={self.car_detector.consecutive_detections}")

                if not self.car_detector.car_in_frame:
                    # Car not in frame, nothing to monitor
                    self.current_state = "waiting_for_car"
                    continue
                
                self.stats["car_detections"] += 1
                self.current_state = "car_detected"

                # Stage 2: Proximity tracking
                nearby_objects = self.proximity_tracker.process_frame(
                    frame,
                    car_bbox=self.car_detector.car_bbox,
                    timestamp=timestamp
                )

                if nearby_objects:
                    self.stats["proximity_events"] += 1
                    self.current_state = "proximity_detected"

                # Stage 3: Contact Classification
                contacts = self.contact_classifier.process_frame(
                    frame, 
                    nearby_objects,
                    poses=None,  # Would come from pose model
                    timestamp=timestamp
                )
                
                if contacts:
                    self._handle_contact_events(contacts, timestamp)
                else:
                    # No contact - if we were recording, check if we should stop
                    if self.recorder.is_recording:
                        time_since_contact = timestamp - self.last_contact_time
                        post_buffer = self.config.get("recording", {}).get("post_buffer_seconds", 5)

                        if time_since_contact > post_buffer:
                            # Capture recording info before stopping
                            recording_path = str(self.recorder._current_recording_path) if self.recorder._current_recording_path else None
                            recording_duration = self.recorder.recording_duration if hasattr(self.recorder, 'recording_duration') else 0
                            contact_type = self.recorder._recording_reason if hasattr(self.recorder, '_recording_reason') else "contact"

                            self.recorder.stop_recording()
                            self.current_state = "monitoring"
                            logger.info("Recording stopped (no contact for post-buffer period)")

                            # Send video via Telegram
                            if self.notifier and recording_path:
                                self.notifier.send_recording_end(
                                    video_path=recording_path,
                                    duration=recording_duration,
                                    contact_type=contact_type
                                )
                
                self.stats["frames_processed"] += 1
                
            except Exception as e:
                logger.exception(f"Error in processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Processing loop ended")
    
    def _capture_frame(self):
        """Capture a frame from the camera."""
        return self.camera.get_frame(timeout=0.1)
    
    def _handle_contact_events(self, contacts: list, timestamp: float):
        """Handle detected contact events."""
        for contact in contacts:
            self.stats["contact_events"] += 1
            self.last_contact_time = timestamp

            logger.info(
                f"Contact detected: {contact.contact_type.value} "
                f"(confidence: {contact.confidence:.2f})"
            )

            # Create event in owner profile for tracking
            event_id = None
            if self.owner_profile:
                recording_path = None
                if self.recorder.is_recording and hasattr(self.recorder, '_current_recording_path'):
                    recording_path = str(self.recorder._current_recording_path)

                event_id = self.owner_profile.create_event(
                    recording_path=recording_path,
                    timestamp=timestamp
                )

            # Start recording if not already
            if not self.recorder.is_recording:
                self.recorder.start_recording(
                    reason=contact.contact_type.value,
                    metadata={
                        "contact_type": contact.contact_type.value,
                        "confidence": contact.confidence,
                        "location": contact.location,
                        "actor_id": contact.actor_id,
                        "event_id": event_id
                    }
                )
                self.stats["recordings_triggered"] += 1
                self.current_state = "recording"
                logger.info(f"Recording started: {contact.contact_type.value}")

                # Update event with recording path
                if self.owner_profile and event_id:
                    event = self.owner_profile.get_event(event_id)
                    if event and hasattr(self.recorder, '_current_recording_path'):
                        event['recording_path'] = str(self.recorder._current_recording_path)

                # Send Telegram notification with snapshot
                if self.notifier:
                    frame_jpeg = None
                    if self._last_frame is not None:
                        _, jpeg_data = cv2.imencode('.jpg', self._last_frame)
                        frame_jpeg = jpeg_data.tobytes()

                    self.notifier.send_recording_start(
                        contact_type=contact.contact_type.value,
                        confidence=contact.confidence,
                        frame_data=frame_jpeg
                    )

            # Log to database
            self._log_event(contact, timestamp)
    
    def _log_event(self, contact, timestamp: float):
        """Log contact event to database."""
        self.database.log_contact_event(
            event_type=contact.contact_type.value,
            confidence=contact.confidence,
            location=contact.location,
            actor_type=None,  # Could be 'person', 'vehicle', etc.
            actor_track_id=contact.actor_id,
            duration=contact.duration,
            recording_path=str(self.recorder._current_recording_path) if self.recorder.is_recording else None,
            metadata={
                "timestamp": timestamp,
                "session_id": self._session_id
            }
        )

    def _on_owner_confirmed(self, event_id: int):
        """
        Callback when user replies 'me' to an alert.

        Adds the event's appearance data to owner profile.
        """
        if not self.owner_profile:
            return

        # Confirm the event (uses latest if event_id is 0)
        event_id_to_use = event_id if event_id > 0 else None

        if self.owner_profile.confirm_owner(event_id=event_id_to_use):
            logger.info("Owner confirmation recorded")

            # Log how many samples we have now
            sample_count = self.owner_profile.sample_count
            if sample_count >= 3:
                logger.info(f"Owner profile trained with {sample_count} samples")
            else:
                logger.info(f"Owner profile has {sample_count}/3 samples needed for recognition")

    def get_status(self) -> dict:
        """Get current pipeline status."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            "state": self.current_state,
            "running": self._running,
            "car_in_frame": self.car_detector.car_in_frame,
            "recording": self.recorder.is_recording if hasattr(self.recorder, 'is_recording') else False,
            "stats": self.stats,
            "uptime_seconds": uptime
        }
