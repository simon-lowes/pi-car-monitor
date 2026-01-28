# Pi Car Monitor - Project Context

## Overview
This project creates a privacy-respecting car security monitoring system using a Raspberry Pi 5 with Hailo AI HAT. The system is designed to ONLY record when there is physical contact with the owner's vehicle, not general activity in the parking area.

## Hardware
- Raspberry Pi 5 (8GB RAM)
- Hailo AI HAT (13 TOPS or 26 TOPS)
- Pi Camera Module (facing car parking spaces)
- 1TB storage

## Owner's Requirements

### Privacy-First Design
- **No continuous recording** of public spaces
- **Only record when**: Someone makes physical contact with the owner's car
- **Ignore**: People walking past, dogs, general activity that doesn't involve the car
- **Purpose**: Capture evidence of accidental or intentional damage for insurance/authorities

### Detection Pipeline (Multi-Stage)
```
Stage 1: Car Recognition
    → Is the owner's specific car in frame? 
    → If no: ignore entirely

Stage 2: Proximity Detection  
    → Is any person/object within contact distance of the car?
    → If no: standby (no recording)

Stage 3: Contact Classification
    → Is there physical contact with the car? (hand touching, leaning, impact)
    → If no: log metadata only, don't record video

Stage 4: Record + Alert
    → Contact detected: start recording, save footage, optional alert
```

### Car Identification Method
The owner wants to bootstrap recognition using:
1. **Known car details**: Make, model, year, colour (to be provided)
2. **Reference images**: Downloaded from web for the specific model
3. **Calibration shots**: A few photos from the actual camera angle
4. **Number plate**: For definitive confirmation (NEVER sent to any API)

### Number Plate Security
- Process plate recognition LOCALLY only using Hailo
- Store only a SHA-256 hash of the plate for matching
- Raw plate text must NEVER leave the device
- No cloud APIs for plate recognition

### Recording Behaviour
- Pre-buffer: Keep rolling 10-second buffer in memory
- On trigger: Save buffer + continue recording until contact ends + 5 seconds
- Auto-delete: Footage older than 30 days unless manually flagged
- Storage: Local only, never cloud-synced

## Technical Stack
- **Inference**: Hailo-8L with HailoRT
- **Detection Models**: YOLOv8 for objects, pose estimation for contact detection
- **Pipeline**: GStreamer with Hailo plugins
- **Language**: Python 3.11+
- **Framework**: Hailo TAPPAS / Hailo Apps Infra

## File Structure
```
pi-car-monitor/
├── CLAUDE.md                 # This file - project context
├── README.md                 # Setup instructions
├── config/
│   ├── config.yaml           # Main configuration
│   └── car_profile.yaml      # Owner's car details (git-ignored)
├── models/
│   ├── download_models.sh    # Fetch base models
│   └── custom/               # Fine-tuned models go here
├── src/
│   ├── main.py               # Entry point
│   ├── car_detector.py       # Stage 1: Car recognition
│   ├── proximity_tracker.py  # Stage 2: Proximity zone
│   ├── contact_classifier.py # Stage 3: Contact detection
│   ├── recorder.py           # Stage 4: Recording logic
│   ├── plate_handler.py      # Secure plate processing
│   └── pipeline.py           # GStreamer pipeline setup
├── training/
│   ├── collect_references.py # Download reference images
│   ├── calibrate.py          # Capture calibration shots
│   └── fine_tune.py          # Transfer learning script
├── utils/
│   ├── hailo_utils.py        # Hailo helper functions
│   └── crypto.py             # Hashing utilities
├── data/
│   ├── reference_images/     # Downloaded car model images
│   ├── calibration/          # Real camera calibration shots
│   └── recordings/           # Triggered recordings (git-ignored)
├── logs/
│   └── events.db             # SQLite event log
├── requirements.txt
├── install.sh                # One-command setup
└── systemd/
    └── car-monitor.service   # Auto-start on boot
```

## Setup Sequence (for Claude Code on Pi)
1. Run `install.sh` to set up dependencies and Hailo drivers
2. Edit `config/car_profile.yaml` with owner's car details
3. Run `training/collect_references.py` to gather reference images
4. Run `training/calibrate.py` to capture real camera shots
5. Run `training/fine_tune.py` to create custom detector
6. Test with `python src/main.py --test`
7. Enable service with `systemctl enable car-monitor`

## Key Decisions
- Using GStreamer for efficient video pipeline (Hailo-optimised)
- SQLite for event logging (lightweight, no server needed)
- YAML for configuration (human-readable)
- Systemd for process management (auto-restart, boot startup)

## Privacy Safeguards Built In
- No network transmission of video or plate data
- Configurable auto-deletion of old footage
- Option to blur non-car regions in saved footage
- All inference runs on-device via Hailo

## Notes for Claude Code
When working on this project:
- Always check Hailo driver status before running inference
- Test pipeline changes with `--dry-run` flag first
- The owner prefers minimal dependencies and simple code
- Prioritise reliability over features
- Keep memory usage low - this runs 24/7

## Git Push Instructions
GitHub CLI (`gh`) is authenticated on this Pi. To push changes:
```bash
gh auth setup-git && git push origin main
```
This configures git to use gh credentials. Do not say you can't push - this works.

---

## Session Change Log

### Session: 2026-01-27 — Critical Detection Accuracy Overhaul

**Problem reported by owner:**
The system was alerting on events from neighboring cars (especially tail lights at night) rather than specifically the owner's car. Alerts were essentially acting as a generic event detector, not a car-specific monitor. Additionally, "null" replies (false positive feedback) were being acknowledged but silently discarded — the system was NOT learning from them despite telling the user it would.

**Root cause analysis:**
1. `_match_model()` in `car_detector.py` was a TODO placeholder returning 0.0 — model/shape matching was completely non-functional
2. Car identification fell through to loose `position_fallback` (any vehicle within 100px of baseline) or `zone_fallback` (any grey-ish car in the broad zone)
3. Departure alerts triggered on ANY lights/motion in the baseline bbox zone, without verifying the activity related to the owner's car
4. `pipeline.py` never called `set_false_positive_callback()` on the notifier, so "null" replies hit a dead code path
5. Grey colour matching (saturation < 60, value 80-180) was extremely broad — matched most neutral-coloured cars

**Changes made:**

#### `src/car_detector.py`
- **Implemented `_match_model()`**: Real histogram + structural similarity matching against 5 baseline snapshot images captured at different lighting conditions. Compares HSV histograms (hue, saturation, value) and grayscale structural similarity via normalized cross-correlation
- **Added `_load_baseline_images()`**: Loads and precomputes features from `data/baselines/baseline_*.jpg` at startup, extracting the car region using the known baseline bbox
- **Added false positive zone tracking**: `record_false_positive()`, `_is_in_false_positive_zone()`, `_load_false_positives()`, `_save_false_positives()` — learns which screen regions produce false positives and suppresses future detections there
- **Tightened `_check_if_target()` identification logic**:
  - Removed standalone position fallback at score >= 0.6 (was too loose)
  - Removed standalone colour-in-zone fallback (grey matches too many cars)
  - Now requires MULTIPLE corroborating signals: model+position, colour+position, or very tight position (>= 0.85) + plausible colour
  - Added false positive zone rejection before identification
- **Stored `_fp_data_path`** at `data/false_positives.yaml` for persistence across restarts

#### `src/pipeline.py`
- **Wired up `set_false_positive_callback()`**: "null" replies now call `_on_false_positive_reported()` which records the detection zone in the car detector's FP learning system
- **Added `_on_false_positive_reported()` method**: Records FP in car detector, marks event in owner profile, logs to database
- **Listener starts even without owner_profile**: FP feedback works regardless of owner recognition state
- **Passes `car_match_reasons`** to presence tracker so it knows HOW the car was identified
- **Suppresses vehicle contact alerts** when car ID quality is "position_only" or "weak" — prevents neighboring car contacts from triggering alerts
- **Stores `_last_car_detection`** for inclusion in Telegram alert messages

#### `src/presence_tracker.py`
- **Added `car_match_reasons` parameter** to `process_frame()`
- **Tracks `_car_positively_identified` and `_frames_positively_identified`**: Distinguishes between strong identification (plate, colour+model) and weak (position-only)
- **Daylight departure suppression**: In daylight, departure signals (lights/motion) are suppressed unless the car was recently positively identified. This prevents neighbor car tail lights from triggering departure alerts. Exception: "car_missing" and "position_shift" signals still work since they indicate the actual car left
- **Night detection unchanged**: At night, light-based departure detection still works as before since it's the only viable method

#### `src/telegram_notifier.py`
- **Expanded false positive reply vocabulary**: Now accepts "nope", "not mine", "other car", "neighbour", "neighbor", "next car", "wrong car", "not my car", "different car", "ignore" in addition to existing terms
- **Updated FP confirmation message**: Now tells user "The detection zone has been logged and will be suppressed in future" instead of the misleading "Will improve detection"
- **Added `car_id_reasons` to `send_recording_start()`**: Alert messages now show HOW the car was identified (e.g. "Car ID: plate_match" or "Car ID: colour_match, position_match")
- **Added reply instructions**: Contact alerts now include "Reply 'null' if this is wrong (teaches the system)"

#### New files created
- `data/false_positives.yaml` — Persisted false positive zone data (created on first FP report)

**Key architectural decisions:**
- Model matching uses OpenCV histogram comparison rather than a neural network — lightweight enough for 24/7 Pi operation
- False positive zones use center-distance clustering with expanding exclusion radius based on report count
- Daylight vs night distinction is critical: in daylight, demand positive car ID; at night, accept baseline zone activity since visual ID is impossible
- Multiple corroborating signals required for car identification (removed all single-factor fallbacks)

**Known remaining limitations:**
- `_match_colour()` for grey is still broad (achromatic colours are inherently hard to distinguish) — partially mitigated by requiring colour + another signal
- Plate recognition depends on camera angle and resolution — may not work consistently
- At night, the system cannot distinguish the owner's car from a neighbor's by appearance, so departure alerts from tail lights in the zone may still occur — this is acknowledged as a hardware limitation
- The 5 baseline images were all captured Jan 24-25; seasonal/weather variation may reduce match quality over time — baseline updates on car return help with this

**Owner's car details (for reference):**
- 2021 Toyota Corolla Estate Icon, grey metallic, roof rails, privacy glass
- Plate: FX71YTS (stored as SHA-256 hash only)
- Baseline position: center (1321, 414) in 1920x1080 frame, bbox 1202-1441 x 361-467
- Parking: appears to be upper-right quadrant of camera view
