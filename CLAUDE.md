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

### Session: 2026-01-31 — Passing Vehicle False Positive Overhaul

**Problem reported by owner:**
Vehicle contact false positives from passing traffic. When a van or car drives past on the road behind the parked car, the 2D bounding box overlap triggers a collision alert. The owner replies "null" but the system wasn't learning the right thing — it was recording the car's own detection zone as an FP, not learning that the OTHER vehicle was just passing through. The owner was unsure if feedback was making any difference.

**Root cause analysis:**
1. Contact detection used purely 2D bounding box IoU with no depth/distance estimation — a van driving past 30 feet behind the car visually overlaps in 2D
2. No motion direction analysis — a vehicle moving laterally at road speed should not trigger a contact alert
3. "null" feedback recorded the car detector's own bbox as an FP zone, not the other vehicle's position. Vehicle contact FPs need completely different handling from car detection FPs
4. FP zones at the baseline position were recorded but never suppressed (by design, to avoid suppressing real detections) — wasting FP slot capacity
5. Telegram response was generic ("zone has been logged") regardless of what type of FP occurred

**Changes made:**

#### `src/contact_classifier.py`
- **Added `_is_vehicle_passing()`**: Detects vehicles with consistent lateral (horizontal) motion. Requires horizontal speed > 8px/frame, predominantly horizontal movement (>2x vertical), and >70% direction consistency. Passing vehicles are filtered out before alerts
- **Added `_is_similar_depth()`**: Pseudo-depth filter using apparent size comparison. If the other vehicle's area is <30% or >300% of the target car's area, they're likely at different distances. Also checks bottom-edge position (ground plane perspective heuristic)
- **Added transit zone learning**: `_load_transit_zones()`, `_save_transit_zones()`, `_is_in_transit_zone()`, `record_vehicle_false_positive()` — learns where vehicles regularly pass through. When user replies "null" to a vehicle contact alert, the OTHER vehicle's position is recorded as a transit zone. Future vehicle contacts from that region are suppressed
- **Added `last_vehicle_contact_info`**: Stores the bbox/details of the most recent vehicle contact alert so the FP callback can record the right thing
- **Added `is_passing` field to `TrackedVehicle`**: Tracks whether each vehicle is classified as passing through
- **Updated `_should_alert_vehicle_contact()`**: Now checks 7 conditions (was 4): overlap threshold, persistence, not stationary, not passing, similar depth, not in transit zone, cooldown
- **New config options**: `passing_speed_threshold`, `size_ratio_min`, `size_ratio_max` in `vehicle_contact` section
- **New data file**: `data/transit_zones.yaml` — persisted transit zone data

#### `src/pipeline.py`
- **Rewired `_on_false_positive_reported()`**: Now checks if the last alert was a vehicle contact (within 5 min). If so, routes to `contact_classifier.record_vehicle_false_positive()` (transit zone learning). Otherwise falls through to `car_detector.record_false_positive()` (detection zone learning). Returns detail string for Telegram message
- **Logs transit zone count** alongside FP zone count in database metadata

#### `src/car_detector.py`
- **Added baseline guard to `record_false_positive()`**: Skips recording FP zones within 50px of the car's baseline position, since those zones are ineffective (the suppression logic always skips detections at baseline) and waste FP capacity. Logs an informative message explaining why

#### `src/telegram_notifier.py`
- **Specific FP confirmation messages**: Now tells the user exactly what was learned:
  - Vehicle contact FP: "passing vehicle recorded as transit zone. Future vehicles in that lane will be filtered out"
  - No data available: "couldn't determine the source of the false alert"
  - Detection FP: "detection zone logged and will be suppressed"
- **Calls callback before sending reply**: Gets the FP detail from the pipeline callback to include in the response

**Key architectural decisions:**
- Passing vehicle detection uses motion vector analysis rather than trying to classify vehicle direction from appearance — computationally cheap and works at any time of day
- Depth estimation uses apparent size ratio as a proxy rather than requiring stereo vision or depth sensors — simple heuristic that handles the most egregious cases (large van in foreground vs small parked car)
- Transit zones are separate from car detection FP zones because they serve different purposes: transit zones suppress VEHICLE CONTACTS in a region, while FP zones suppress CAR DETECTION in a region
- Bottom-edge comparison uses 25% frame height tolerance — generous enough to handle different vehicle heights while still filtering vehicles clearly in the foreground or background

### Session: 2026-02-01 — DEPARTING State Stuck Bug Fix

**Problem reported by owner:**
Owner left in their car and returned during daylight, but received no departure or return alerts. The system appeared to not detect the trip at all.

**Root cause analysis:**
The presence tracker entered DEPARTING state at 09:47 AM due to departure signals (motion/lights) but:
1. The car was still continuously detected (`car_in_frame=True`) — so the ABSENT transition (requires car missing for 30 frames) never triggered
2. The position confidence never consistently exceeded 0.8 for 5 frames — so the back-to-PRESENT "departure cancelled" transition never triggered either
3. **There was no timeout on the DEPARTING state** — so it stayed stuck in DEPARTING for 9+ hours
4. When the owner actually left at ~16:10 and returned ~17:00, the system was already in DEPARTING and couldn't detect a new departure or the return cycle

**Changes made:**

#### `src/presence_tracker.py`
- **Added `departing_timeout_seconds` config** (default 180s / 3 minutes): If the car is continuously detected for longer than this while in DEPARTING state, it clearly didn't leave. The system reverts to PRESENT and updates the baseline if in daylight
- **Timeout uses wall-clock time** via `departure_started_at` timestamp — not frame counts, so it's independent of frame rate or processing load

#### `config/config.yaml`
- **Added `departing_timeout_seconds: 180`** to presence_tracking section

### Session: 2026-02-02 — Departure Alert Spam + Impact FP Fixes

**Problems reported by owner:**
1. Massive departure alert spam — 43 false departure alerts in one day, all timing out (car never actually left)
2. 3 false impact/collision recordings at 16:05, 16:23, and 16:24 (confidences 0.58, 0.62, 0.65)
3. "null" replies to impact FPs (~20 replies) were all mis-routed to car detector FP handler (which rejected them as overlapping baseline), so feedback had zero effect

**Root cause analysis:**

1. **Departure spam:** The 180s DEPARTING timeout (added Feb 1) correctly reverted to PRESENT, but there was NO cooldown — departure signals immediately re-accumulated and the system re-entered DEPARTING within seconds. This repeated every ~3 minutes indefinitely.

2. **Impact FPs:** The impact detection threshold was too low (15% of car zone pixels changing) with no minimum confidence filter. Motion ratios of 0.19-0.22 triggered alerts — likely caused by shadows, light changes, or nearby movement.

3. **FP routing:** `_on_false_positive_reported()` in pipeline.py only checked `last_vehicle_contact_info` (set by vehicle-to-vehicle contacts). Impact events are generated by `_check_impact_event()` which never set this field. So impact "null" replies fell through to `car_detector.record_false_positive()` which correctly identified them as overlapping baseline and discarded them — the user's feedback was silently wasted.

**Changes made:**

#### `src/presence_tracker.py`
- **Added departure cooldown after timeout** (`departure_cooldown_after_timeout`, default 1800s / 30 min): After a DEPARTING state times out or is cancelled back to PRESENT, departure signals are suppressed for the cooldown period. This prevents the PRESENT→DEPARTING→timeout→PRESENT→DEPARTING spam cycle.
- **Cooldown tracks via `_last_departing_timeout_at` timestamp**: Set on timeout and departure_cancelled, cleared when car actually leaves (ABSENT state) so real departures aren't blocked.
- **Cooldown check** inserted before the DEPARTING transition in `_handle_present_state()`

#### `src/contact_classifier.py`
- **Raised impact motion threshold** from 0.15 (15%) to 0.25 (25%) — configurable via `detection.impact.motion_ratio_threshold`
- **Added minimum confidence filter** for impacts: default 0.65, configurable via `detection.impact.min_confidence`. Events below this are logged at debug level and discarded.
- Today's false impacts (0.58, 0.62, 0.65 confidence from motion ratios 0.19-0.22) would ALL be filtered by the raised motion threshold (requires 0.25+)

#### `src/pipeline.py`
- **Added `_last_contact_alert_info` tracking**: Stores contact type, confidence, location, actor_id, and timestamp for ALL contact alerts (not just vehicle contacts)
- **Rewrote `_on_false_positive_reported()` routing**: Now checks `_last_contact_alert_info` first (within 5 min), routes by contact type:
  - `vehicle` → transit zone learning (existing behavior)
  - `impact` → acknowledged + logged (impact FPs are motion-based, can't learn a zone from them)
  - `hand_touch`/`body_lean`/etc → acknowledged + logged
  - Falls back to vehicle contact info check, then car detector FP zones

#### `src/telegram_notifier.py`
- **Expanded FP confirmation messages**: Now handles impact_fp ("false impact alert noted, sensitivity tightened"), person contact FPs, and generic contact FPs with specific messages instead of the generic "detection zone logged" that was confusing and inaccurate for non-detection FPs

#### `config/config.yaml`
- **Added `departure_cooldown_after_timeout: 1800`** to presence_tracking section
- **Added `detection.impact` section**: `motion_ratio_threshold: 0.25`, `min_confidence: 0.65`

**Expected impact:**
- Departure alert spam: reduced from 43/day to at most 1-2 (one alert, then 30 min cooldown)
- Impact FPs: all 3 of today's false impacts would be filtered by the raised threshold
- FP feedback: "null" replies to impact alerts now acknowledged correctly instead of silently wasted

**Service restarted:** 17:47 UTC, running from UNKNOWN state

**Automation setup:**
- Created `scripts/nightly-analyse.sh` — runs Claude Code headless at 3 AM via cron
- Uses `script -qc` wrapper to work around TTY bug in `claude -p`
- Agent reads journalctl logs, FP/transit zone data, config thresholds
- Can make safe code fixes (threshold adjustments, new filters) autonomously
- Logs to `logs/nightly/analysis_YYYYMMDD_HHMMSS.log`
- Lock file prevents overlapping runs, 10-minute timeout, 25 max turns
- Tested end-to-end on this Pi — headless tool use confirmed working

**Hardware notes:**
- Two cameras available: Camera 0 (imx708_wide, standard) and Camera 1 (imx708_wide_noir, NoIR)
- Both point through window, no IR illuminator. NoIR still gathers more ambient light at night.
- Only Camera 0 currently used. NoIR camera is a future improvement opportunity.

**Known remaining issues:**
- "Event loop is closed" error on ~50% of Telegram sends (non-critical, pre-existing)
- Owner recognition has very few samples (0-2 "me" replies) — not yet reliable
- Video sending still fails with the same event loop error
