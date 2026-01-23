# Changelog

All notable changes to the Pi Car Monitor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

#### Phase 1: Car Presence State Machine
- New `presence_tracker.py` module with 5-state machine:
  - UNKNOWN → PRESENT → DEPARTING → ABSENT → RETURNING
- Dynamic car position baseline establishment and persistence
- Baseline stored in `data/car_baseline.yaml` with position tolerance
- Baseline snapshots saved for reference
- State transitions trigger appropriate alerts

#### Phase 2: Low-Light Departure Detection
- New `low_light_detector.py` module for detecting departures in darkness
- Enhanced light condition detection (DAYLIGHT, TWILIGHT, LOW_LIGHT, NIGHT)
- Vehicle light detection:
  - Tail lights, brake lights, reverse lights
  - Headlights, interior dome lights, indicators
  - Paired light detection for higher confidence
  - Light persistence filtering to reduce false positives
- Motion analysis with optical flow:
  - Direction detection (toward_camera = backing out)
  - Car-sized motion filtering
- Light pattern recognition (parked, running, reversing, braking, door_open)

#### Phase 3: Owner Identification at Departure
- New `person_features.py` module for appearance-based recognition
- Feature extraction from person detections:
  - Upper/lower body color histograms (HSV for lighting invariance)
  - Dominant colors via k-means clustering
  - Body proportions from pose keypoints
  - Combined feature vector for matching
- `DepartureActorTracker` class to identify who is getting into the car
- Owner matching against stored profile during departure
- Alert urgency levels based on owner confidence:
  - LOW: Likely owner (≥80% match)
  - MEDIUM: Possibly owner (60-80% match)
  - HIGH: Unknown person (<60% match)
- Departure snapshots saved for training

#### Phase 4: Temporal Pattern Learning
- New `temporal_patterns.py` module for learning usage patterns
- `DepartureEvent` records with timestamps, owner confirmation, duration
- `DeparturePattern` for recurring patterns (e.g., "Fri 16:30-17:30")
- Pattern clustering algorithm (30-minute time windows)
- Confidence scoring:
  - Base confidence + per-occurrence bonus
  - Decay for patterns not seen in 30+ days
- Pattern matching to identify expected departures
- Alert suppression for high-confidence patterns with owner match
- Duration tracking and statistics
- YAML persistence in `data/temporal_patterns.yaml`

### Changed
- `pipeline.py`: Integrated all new modules
  - Early person/pose detection for departure tracking
  - Presence state handling with pattern matching
  - Enhanced departure alerts with owner and pattern info
  - Return time recording for duration learning
  - Owner confirmation now records departure for pattern learning
- `config/config.yaml`: Added new configuration sections
  - `presence_tracking`: Baseline and departure detection settings
  - `temporal_patterns`: Pattern learning and alert suppression settings

### Configuration Options Added
```yaml
presence_tracking:
  enabled: true
  baseline_stability_frames: 10
  position_tolerance_px: 50
  area_tolerance_pct: 0.15
  departure_confirmation_frames: 5
  absence_confirmation_frames: 30
  enable_light_detection: true
  light_persistence_frames: 3

temporal_patterns:
  enabled: true
  min_occurrences_for_pattern: 2
  time_cluster_window_minutes: 30
  pattern_decay_days: 30
  suppress_alerts_after_confirmations: 10
  suppress_confidence_threshold: 0.85
```

---

## [0.1.0] - 2026-01-19

### Added
- Initial release with core functionality
- Car detection using YOLOv8 on Hailo AI HAT
- Proximity tracking for objects near the car
- Contact classification (hand touch, body lean, vehicle contact, impact)
- Circular buffer recording with pre/post buffers
- Telegram notifications with photo and video
- Owner recognition via "me" reply training
- SQLite event logging
- Systemd service for auto-start
