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
