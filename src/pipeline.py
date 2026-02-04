"""
Car Monitor Pipeline
====================
Orchestrates the multi-stage detection pipeline:
1. Car Detection → 2. Proximity Tracking → 3. Contact Classification → 4. Recording

Uses GStreamer with Hailo plugins for efficient video processing.
"""

import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import queue

import cv2
import numpy as np

from camera import ThreadedCamera
from car_detector import CarDetector
from proximity_tracker import ProximityTracker
from contact_classifier import ContactClassifier, ContactType
from recorder import CircularRecorder
from database import EventDatabase
from utils.hailo_utils import HailoDevice, postprocess_yolov8_pose
from telegram_notifier import TelegramNotifier, create_notifier_from_config
from owner_profile import OwnerProfile, create_profile_from_config
from presence_tracker import PresenceTracker, PresenceState
from temporal_patterns import TemporalPatternDB, create_pattern_db_from_config

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

        # Initialize presence tracker for departure/return detection
        presence_config = config.get('presence_tracking', {})
        self.presence_tracking_enabled = presence_config.get('enabled', False)
        if self.presence_tracking_enabled:
            baseline_path = presence_config.get(
                'baseline_path',
                'data/car_baseline.yaml'
            )
            self.presence_tracker = PresenceTracker(config, baseline_path)
            logger.info(f"Presence tracking enabled, state={self.presence_tracker.state.name}")
        else:
            self.presence_tracker = None
            logger.info("Presence tracking disabled")

        # Initialize temporal pattern learning
        self.pattern_db = create_pattern_db_from_config(config)
        if self.pattern_db:
            logger.info(f"Temporal patterns enabled ({self.pattern_db.get_status()['pattern_count']} patterns)")
        else:
            logger.info("Temporal pattern learning disabled")

        # Store last frame for snapshot
        self._last_frame = None

        # Track last contact alert for FP feedback routing
        self._last_contact_alert_info: Optional[dict] = None
        
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
            "presence_state_changes": 0,
            "departures_detected": 0,
            "returns_detected": 0,
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

        # Start Telegram reply listener for owner recognition AND false positives
        if self.notifier:
            owner_cb = self._on_owner_confirmed if self.owner_profile else None
            # Always start listener - even without owner profile, we need FP feedback
            self.notifier.start_listener(owner_cb or (lambda eid: None))
            self.notifier.set_false_positive_callback(self._on_false_positive_reported)

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
                if car_detection and car_detection.is_target_car:
                    self._last_car_detection = car_detection

                # Early person/pose detection for departure tracking
                # Run this before presence tracking so we can identify who's getting in the car
                early_person_detections = None
                early_pose_detections = None

                if self.presence_tracking_enabled and self.presence_tracker:
                    # Only run early detection if in relevant states and car zone known
                    presence_state = self.presence_tracker.state
                    if presence_state in [PresenceState.PRESENT, PresenceState.DEPARTING]:
                        if self.presence_tracker.baseline:
                            # Run person detection using proximity tracker's detector
                            try:
                                from utils.hailo_utils import postprocess_yolov8_detections
                                outputs = self.hailo.run_inference('detector', frame)
                                if outputs:
                                    frame_h, frame_w = frame.shape[:2]
                                    all_detections = postprocess_yolov8_detections(
                                        outputs,
                                        conf_threshold=0.5,
                                        orig_shape=(frame_h, frame_w)
                                    )
                                    # Filter to persons only
                                    early_person_detections = [
                                        d for d in all_detections if d.get('class') == 'person'
                                    ]

                                    # Run pose if persons found
                                    if early_person_detections and 'pose' in self.hailo.models:
                                        pose_outputs = self.hailo.run_inference('pose', frame)
                                        if pose_outputs:
                                            early_pose_detections = postprocess_yolov8_pose(
                                                pose_outputs,
                                                conf_threshold=0.5,
                                                orig_shape=(frame_h, frame_w)
                                            )
                            except Exception as e:
                                logger.debug(f"Early person detection failed: {e}")

                # Stage 0 (runs after detection): Presence Tracking
                presence_result = None
                if self.presence_tracking_enabled and self.presence_tracker:
                    # Pass match reasons so presence tracker knows HOW the car was identified
                    match_reasons = None
                    if car_detection and car_detection.is_target_car:
                        match_reasons = car_detection.match_reasons

                    presence_result = self.presence_tracker.process_frame(
                        frame=frame,
                        car_detected=self.car_detector.car_in_frame,
                        car_bbox=self.car_detector.car_bbox,
                        car_confidence=car_detection.confidence if car_detection else 0.0,
                        car_stable=self.car_detector.detection_stable,
                        person_detections=early_person_detections,
                        pose_detections=early_pose_detections,
                        car_match_reasons=match_reasons
                    )

                    # Sync car_detector baseline if presence_tracker updated it
                    if self.presence_tracker.baseline_updated:
                        self.car_detector.reload_baseline()
                        self.presence_tracker.baseline_updated = False

                    # Handle presence state changes
                    if presence_result.get('state_changed'):
                        self.stats["presence_state_changes"] += 1
                        new_state = presence_result['state']

                        if new_state == PresenceState.DEPARTING:
                            self.stats["departures_detected"] += 1
                            logger.info("Car departure detected!")

                            # Check if we can identify the owner
                            owner_confidence = self._check_owner_at_departure(presence_result)
                            presence_result['owner_match_confidence'] = owner_confidence

                            # Check for temporal pattern match
                            pattern_match = None
                            if self.pattern_db:
                                is_expected, pattern = self.pattern_db.is_expected_departure(datetime.now())
                                if is_expected and pattern:
                                    pattern_match = pattern
                                    presence_result['pattern_match'] = pattern
                                    logger.info(f"Departure matches pattern: {pattern}")

                            # Determine if alert should be suppressed
                            should_suppress = False
                            if pattern_match:
                                temporal_config = self.config.get('temporal_patterns', {})
                                suppress_threshold = temporal_config.get('suppress_confidence_threshold', 0.85)
                                min_confirmations = temporal_config.get('suppress_alerts_after_confirmations', 10)

                                if (pattern_match.confidence >= suppress_threshold and
                                    pattern_match.occurrence_count >= min_confirmations):
                                    # Also need high owner confidence to suppress
                                    if owner_confidence and owner_confidence >= 0.7:
                                        should_suppress = True
                                        logger.info("Alert suppressed (expected pattern + owner identified)")

                            # Send departure alert via Telegram
                            if self.notifier and presence_result.get('should_alert') and not should_suppress:
                                self._send_departure_alert(frame, presence_result)
                            elif should_suppress:
                                logger.info("Departure alert suppressed due to learned pattern")

                        elif new_state == PresenceState.ABSENT:
                            logger.info("Car confirmed absent")

                        elif new_state == PresenceState.RETURNING:
                            logger.info("Car returning!")

                        elif new_state == PresenceState.PRESENT:
                            if presence_result.get('alert_reason') == 'car_returned':
                                self.stats["returns_detected"] += 1
                                logger.info("Car has returned and parked")

                                # Record return time for pattern learning
                                if self.pattern_db:
                                    duration = self.pattern_db.record_return(datetime.now())
                                    if duration:
                                        logger.info(f"Trip duration recorded: {duration} minutes")

                                # Send return alert via Telegram
                                if self.notifier and presence_result.get('should_alert'):
                                    self._send_return_alert(frame, presence_result)

                # Log status every 150 frames (~10 seconds at 15fps)
                if self.frame_count % 150 == 0:
                    presence_state = self.presence_tracker.state.name if self.presence_tracker else "disabled"
                    logger.info(f"Status: state={self.current_state}, presence={presence_state}, "
                               f"frames={self.frame_count}, car_in_frame={self.car_detector.car_in_frame}, "
                               f"detections={self.car_detector.consecutive_detections}")

                if not self.car_detector.car_in_frame:
                    # Car not in frame, nothing to monitor for contact
                    # But presence tracker may still be working (ABSENT/RETURNING states)
                    self.current_state = "waiting_for_car"
                    continue

                self.stats["car_detections"] += 1
                self.current_state = "car_detected"

                # Only proceed with contact detection if we have a positively
                # identified car (not just position fallback), OR if it's dark
                # and we're relying on baseline position
                car_id_quality = "unknown"
                if car_detection and car_detection.match_reasons:
                    reason_names = {r.split(':')[0] for r in car_detection.match_reasons}
                    if reason_names & {'plate_match', 'colour_match', 'model_match',
                                      'model_position_match', 'colour_position_match'}:
                        car_id_quality = "positive"
                    elif 'position_fallback' in reason_names:
                        car_id_quality = "position_only"
                    else:
                        car_id_quality = "weak"

                # Stage 2: Proximity tracking
                nearby_objects = self.proximity_tracker.process_frame(
                    frame,
                    car_bbox=self.car_detector.car_bbox,
                    timestamp=timestamp
                )

                if nearby_objects:
                    self.stats["proximity_events"] += 1
                    self.current_state = "proximity_detected"

                # Run pose estimation if persons are nearby
                poses = None
                persons_nearby = [obj for obj in nearby_objects if obj.get('class') == 'person']
                if persons_nearby and 'pose' in self.hailo.models:
                    try:
                        pose_outputs = self.hailo.run_inference('pose', frame)
                        if pose_outputs:
                            frame_h, frame_w = frame.shape[:2]
                            poses = postprocess_yolov8_pose(
                                pose_outputs,
                                conf_threshold=0.5,
                                orig_shape=(frame_h, frame_w)
                            )
                            if poses:
                                logger.debug(f"Pose estimation: {len(poses)} person(s) detected")
                    except Exception as e:
                        logger.warning(f"Pose estimation failed: {e}")

                # Stage 3: Contact Classification
                contacts = self.contact_classifier.process_frame(
                    frame,
                    nearby_objects,
                    poses=poses,
                    timestamp=timestamp
                )
                
                if contacts:
                    # Suppress ALL contact alerts when car ID is weak
                    # (prevents alerts from neighboring cars being touched)
                    if car_id_quality == "position_only" or car_id_quality == "weak":
                        logger.info(
                            f"Suppressing {len(contacts)} contact alert(s): car ID quality "
                            f"is '{car_id_quality}' (likely contact with neighboring car, not ours)"
                        )
                        contacts = []  # Suppress all contacts when car ID is unreliable

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

                # Extract appearance features from the person who made contact
                person_features = None
                if self._last_frame is not None and contact.actor_id is not None:
                    # Try to get person bbox from nearby objects or pose detections
                    person_bbox = self._get_person_bbox_for_contact(contact)
                    if person_bbox:
                        from owner_profile import extract_person_features
                        person_features = extract_person_features(self._last_frame, person_bbox)
                        if person_features is not None:
                            logger.debug(f"Extracted appearance features for contact event")

                event_id = self.owner_profile.create_event(
                    recording_path=recording_path,
                    timestamp=timestamp,
                    features=person_features
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

                # Track this contact for FP feedback routing
                self._last_contact_alert_info = {
                    'contact_type': contact.contact_type.value,
                    'confidence': contact.confidence,
                    'location': contact.location,
                    'actor_id': contact.actor_id,
                    'timestamp': timestamp,
                }

                # Send Telegram notification with snapshot
                if self.notifier:
                    frame_jpeg = None
                    if self._last_frame is not None:
                        _, jpeg_data = cv2.imencode('.jpg', self._last_frame)
                        frame_jpeg = jpeg_data.tobytes()

                    # Include car ID info in alert
                    id_reasons = []
                    if hasattr(self, '_last_car_detection') and self._last_car_detection:
                        id_reasons = self._last_car_detection.match_reasons

                    self.notifier.send_recording_start(
                        contact_type=contact.contact_type.value,
                        confidence=contact.confidence,
                        frame_data=frame_jpeg,
                        car_id_reasons=id_reasons
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
        Records the departure for temporal pattern learning.
        """
        # Confirm owner in profile
        if self.owner_profile:
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

        # Record departure for temporal pattern learning
        if self.pattern_db:
            # Get owner confidence from the last departure actor if available
            owner_confidence = None
            light_conditions = None

            if self.presence_tracker and self.presence_tracker.last_departure_actor:
                actor = self.presence_tracker.last_departure_actor
                if actor.features:
                    owner_confidence = actor.features.confidence

            # Record the departure as confirmed owner
            self.pattern_db.record_departure(
                timestamp=datetime.now(),
                was_owner=True,
                owner_confidence=owner_confidence,
                light_conditions=light_conditions
            )
            logger.info("Departure recorded for pattern learning")

    def _on_false_positive_reported(self, event_id: int):
        """
        Callback when user replies 'null'/'false' to an alert.

        Routes feedback to the appropriate subsystem based on the last alert type:
        - Vehicle contact FPs → contact_classifier (transit zone learning)
        - Impact FPs → logged (impact threshold already handles sensitivity)
        - Person contact FPs → logged
        - Departure FPs → logged (cooldown handles repeat suppression)
        - Car detection FPs → car_detector (FP zone learning)
        """
        logger.info(f"False positive reported (event_id={event_id})")

        fp_detail = "zone_recorded"

        # Check what the last alert was (contact alerts take priority)
        # No time limit — user may reply days later, feedback is still valid
        last_contact = self._last_contact_alert_info
        if last_contact:
            contact_type = last_contact.get('contact_type', '')

            if contact_type == 'vehicle':
                # Vehicle contact FP → transit zone learning
                fp_detail = self.contact_classifier.record_vehicle_false_positive()
                logger.info(f"Vehicle contact FP recorded: {fp_detail}")
            elif contact_type == 'impact':
                # Impact FP — can't learn a zone from motion-based detection,
                # but log it so we know the threshold may need adjusting
                fp_detail = "impact_fp_acknowledged"
                logger.info(
                    f"Impact FP acknowledged (confidence was "
                    f"{last_contact.get('confidence', '?'):.2f})"
                )
            else:
                # Person contact FP (hand_touch, body_lean, etc.)
                fp_detail = f"{contact_type}_fp_acknowledged"
                logger.info(f"Contact FP acknowledged: {contact_type}")
        else:
            # No contact alert recorded — check vehicle contact info as fallback
            last_vehicle_info = self.contact_classifier.last_vehicle_contact_info
            if last_vehicle_info:
                fp_detail = self.contact_classifier.record_vehicle_false_positive()
                logger.info(f"Vehicle contact FP recorded: {fp_detail}")
            else:
                # Not a contact alert — record as car detection FP zone
                last_bbox = self.car_detector.car_bbox
                self.car_detector.record_false_positive(last_bbox)
                logger.info("Car detection FP zone recorded")

        # If we have an event in owner profile, mark it as not-owner
        if self.owner_profile:
            event = None
            if event_id and event_id > 0:
                event = self.owner_profile.get_event(event_id)
            else:
                event = self.owner_profile.get_latest_event()

            if event:
                event['confirmed_owner'] = False
                event['false_positive'] = True
                logger.info(f"Event {event.get('id', '?')} marked as false positive")

        # Log to database for analysis
        self.database.log_contact_event(
            event_type='false_positive_reported',
            confidence=0.0,
            location=(0, 0),
            actor_type='user_feedback',
            actor_track_id=None,
            duration=0.0,
            recording_path=None,
            metadata={
                'event_id': event_id,
                'session_id': self._session_id,
                'fp_detail': fp_detail,
                'fp_zones_count': len(self.car_detector._false_positive_zones),
                'transit_zones_count': len(self.contact_classifier._transit_zones),
            }
        )

        # Return detail for Telegram response
        return fp_detail

    def _get_person_bbox_for_contact(self, contact) -> Optional[tuple]:
        """
        Get the bounding box of the person who made contact.

        Searches through tracked nearby objects to find one matching the contact's actor_id.

        Returns:
            (x1, y1, x2, y2) tuple or None if not found
        """
        try:
            # Check if proximity tracker has the nearby object
            if hasattr(self, 'proximity_tracker') and self.proximity_tracker:
                nearby = getattr(self.proximity_tracker, 'nearby_objects', [])
                for obj in nearby:
                    if hasattr(obj, 'track_id') and obj.track_id == contact.actor_id:
                        if hasattr(obj, 'bbox') and obj.bbox:
                            return tuple(obj.bbox)

            # Fallback: try to find person from pose detections
            if hasattr(self, '_last_pose_detections') and self._last_pose_detections:
                for pose in self._last_pose_detections:
                    if hasattr(pose, 'bbox') and pose.bbox:
                        return tuple(pose.bbox)

            # Last resort: use contact location to estimate a person region
            if contact.location:
                lx, ly = contact.location
                # Estimate a person-sized bbox around the contact point
                # Typical person is ~50-100 pixels wide, 100-200 tall at this distance
                return (int(lx - 50), int(ly - 150), int(lx + 50), int(ly + 50))

        except Exception as e:
            logger.warning(f"Failed to get person bbox: {e}")

        return None

    def _check_owner_at_departure(self, presence_result: dict) -> Optional[float]:
        """
        Check if the departing actor matches the owner profile.

        Returns owner match confidence or None if no match attempted.
        """
        if not self.owner_profile or not self.owner_profile.is_trained:
            return None

        departure_actor = presence_result.get('departure_actor')
        if not departure_actor or not departure_actor.features:
            return None

        feature_vector = departure_actor.features.feature_vector
        if feature_vector is None:
            return None

        try:
            confidence = self.owner_profile.match(feature_vector)
            is_owner = confidence >= self.owner_profile.confidence_threshold

            if is_owner:
                logger.info(f"Owner identified at departure (confidence: {confidence:.2f})")
            else:
                logger.info(f"Unknown person at departure (confidence: {confidence:.2f})")

            # Store actor data for potential training if user confirms
            if self.owner_profile and departure_actor:
                event_id = self.owner_profile.create_event(
                    features=feature_vector,
                    timestamp=departure_actor.first_seen
                )
                # Store snapshot path if we have one
                if hasattr(departure_actor, 'snapshot') and departure_actor.snapshot is not None:
                    try:
                        snapshot_dir = Path("data/departure_snapshots")
                        snapshot_dir.mkdir(parents=True, exist_ok=True)
                        snapshot_path = snapshot_dir / f"departure_{event_id}_{int(time.time())}.jpg"
                        cv2.imwrite(str(snapshot_path), departure_actor.snapshot)
                        event = self.owner_profile.get_event(event_id)
                        if event:
                            event['snapshot_path'] = str(snapshot_path)
                    except Exception as e:
                        logger.debug(f"Failed to save departure snapshot: {e}")

            return confidence

        except Exception as e:
            logger.error(f"Owner check failed: {e}")
            return None

    def _send_departure_alert(self, frame: np.ndarray, presence_result: dict):
        """Send a departure alert via Telegram."""
        if not self.notifier:
            return

        try:
            # Prepare snapshot
            frame_jpeg = None
            if frame is not None:
                _, jpeg_data = cv2.imencode('.jpg', frame)
                frame_jpeg = jpeg_data.tobytes()

            # Build message
            signals = presence_result.get('departure_signals', [])
            signal_types = [s.signal_type for s in signals] if signals else ['unknown']
            light_conditions = presence_result.get('light_conditions', 'unknown')

            # Check owner identification
            owner_confidence = presence_result.get('owner_match_confidence')
            departure_actor = presence_result.get('departure_actor')
            pattern_match = presence_result.get('pattern_match')

            # Determine alert urgency based on owner match and pattern
            if owner_confidence is not None:
                if owner_confidence >= 0.8:
                    urgency = "LOW"
                    owner_status = f"Likely you ({owner_confidence:.0%})"
                elif owner_confidence >= 0.6:
                    urgency = "MEDIUM"
                    owner_status = f"Possibly you ({owner_confidence:.0%})"
                else:
                    urgency = "HIGH"
                    owner_status = f"Unknown ({owner_confidence:.0%})"
            else:
                urgency = "MEDIUM"
                if departure_actor:
                    owner_status = "Person detected, not trained"
                else:
                    owner_status = "No person identified"

            # Pattern info
            pattern_status = None
            if pattern_match:
                # Pattern reduces urgency
                if urgency == "HIGH":
                    urgency = "MEDIUM"
                day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][pattern_match.day_of_week]
                start = pattern_match.time_window_start.strftime('%H:%M')
                end = pattern_match.time_window_end.strftime('%H:%M')
                pattern_status = f"Matches {day} {start}-{end} ({pattern_match.occurrence_count}x)"

            # Format message based on urgency
            if urgency == "HIGH":
                header = "ALERT: Unknown person departing!"
            elif urgency == "LOW":
                header = "Car departure detected"
            else:
                header = "Car departure detected"

            # Build message
            lines = [header, ""]
            lines.append(f"Signals: {', '.join(signal_types)}")
            lines.append(f"Light: {light_conditions}")
            lines.append(f"Driver: {owner_status}")

            if pattern_status:
                lines.append(f"Pattern: {pattern_status}")

            lines.append("")
            lines.append("Reply 'me' if this was you.")

            message = "\n".join(lines)

            # Send via notifier
            self.notifier.send_alert(
                message=message,
                image_data=frame_jpeg
            )
            logger.info(f"Departure alert sent (urgency: {urgency})")

        except Exception as e:
            logger.error(f"Failed to send departure alert: {e}")

    def _send_return_alert(self, frame: np.ndarray, presence_result: dict):
        """Send a return alert via Telegram."""
        if not self.notifier:
            return

        try:
            # Prepare snapshot
            frame_jpeg = None
            if frame is not None:
                _, jpeg_data = cv2.imencode('.jpg', frame)
                frame_jpeg = jpeg_data.tobytes()

            # Build message
            light_conditions = presence_result.get('light_conditions', 'unknown')
            baseline_updated = presence_result.get('baseline', {}) is not None

            message = (
                f"Car has returned!\n\n"
                f"Light conditions: {light_conditions}\n"
                f"Baseline updated: {'Yes' if baseline_updated else 'No'}\n\n"
                f"Reply 'me' if this was you."
            )

            # Send via notifier
            self.notifier.send_alert(
                message=message,
                image_data=frame_jpeg
            )
            logger.info("Return alert sent")

        except Exception as e:
            logger.error(f"Failed to send return alert: {e}")

    def get_status(self) -> dict:
        """Get current pipeline status."""
        uptime = None
        if self.stats["start_time"]:
            uptime = (datetime.now() - self.stats["start_time"]).total_seconds()

        # Get presence tracking status
        presence_status = None
        if self.presence_tracker:
            presence_status = self.presence_tracker.get_status()

        return {
            "state": self.current_state,
            "running": self._running,
            "car_in_frame": self.car_detector.car_in_frame,
            "recording": self.recorder.is_recording if hasattr(self.recorder, 'is_recording') else False,
            "stats": self.stats,
            "uptime_seconds": uptime,
            "presence_tracking": presence_status
        }

    def save_debug_snapshot(self):
        """
        Save a debug snapshot with detection boxes overlaid.

        Can be triggered via SIGUSR1: kill -USR1 <pid>
        """
        if self._last_frame is None:
            logger.warning("No frame available for debug snapshot")
            return None

        try:
            frame = self._last_frame.copy()

            # Draw baseline box (yellow)
            if self.presence_tracker and self.presence_tracker.baseline:
                baseline = self.presence_tracker.baseline
                x1, y1, x2, y2 = baseline['bbox']['x1'], baseline['bbox']['y1'], baseline['bbox']['x2'], baseline['bbox']['y2']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "baseline", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Draw current detection (green if target car, red otherwise)
            if hasattr(self, '_last_car_detection') and self._last_car_detection:
                det = self._last_car_detection
                x1, y1, x2, y2 = det.bbox
                color = (0, 255, 0) if det.is_target_car else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                # Label
                if det.is_target_car:
                    label = f"YOUR CAR {det.confidence:.0%}"
                    cv2.rectangle(frame, (x1, y1 - 35), (x1 + 250, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

                    # Match reasons
                    reasons = ", ".join(det.match_reasons[:3])  # Limit to 3
                    cv2.putText(frame, reasons, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Add timestamp and status
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            status = f"State: {self.current_state} | Car: {'YES' if self.car_detector.car_in_frame else 'NO'}"
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Save
            debug_dir = Path("data/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            output_path = debug_dir / f"live_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(output_path), frame)

            logger.info(f"Debug snapshot saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save debug snapshot: {e}")
            return None

    def _setup_signal_handlers(self):
        """Set up signal handlers for debug features."""
        def sigusr1_handler(signum, frame):
            logger.info("SIGUSR1 received - saving debug snapshot")
            self.save_debug_snapshot()

        signal.signal(signal.SIGUSR1, sigusr1_handler)
        logger.info("Signal handlers configured (SIGUSR1 = save debug snapshot)")
