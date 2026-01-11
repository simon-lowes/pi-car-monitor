# Pi Car Monitor 🚗🔒

A privacy-respecting car security system for Raspberry Pi 5 with Hailo AI acceleration.

## What It Does

This system monitors your parked car and **only records when someone makes physical contact with it**. It's designed to capture evidence of damage (accidental or intentional) while respecting the privacy of passersby.

### What Triggers Recording ✅
- Someone touching your car with their hand
- Someone leaning on your car
- Another vehicle making contact with your car
- Impact/collision events

### What Does NOT Trigger Recording ❌
- People walking past your car
- Someone walking their dog past
- General activity in the parking area
- Vehicles passing by without contact

## Privacy First Design

- **No continuous recording** - Only records contact events
- **Local processing only** - All AI runs on-device via Hailo
- **Plate data never transmitted** - Your plate is stored as a hash only
- **Auto-deletion** - Old footage is automatically deleted
- **Purpose-limited** - Designed solely for damage evidence

## Hardware Requirements

- Raspberry Pi 5 (8GB recommended)
- Hailo AI HAT (8L or 8)
- Pi Camera Module (facing your parking space)
- Storage (1TB SSD recommended for extended recording)

## Quick Start

```bash
# Clone the project
git clone <your-repo-url> ~/pi-car-monitor
cd ~/pi-car-monitor

# Run installation
chmod +x install.sh
./install.sh

# Configure your car details
nano config/car_profile.yaml

# Run calibration
source venv/bin/activate
python src/main.py --calibrate

# Start monitoring
python src/main.py
```

## Using with Claude Code

This project is designed to be extended with Claude Code. After installing Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
cd ~/pi-car-monitor
claude
```

Claude will read `CLAUDE.md` and understand the full project context, allowing you to:
- Fine-tune detection thresholds
- Add new detection features
- Debug issues
- Extend functionality

## Configuration

### Main Config (`config/config.yaml`)
- Camera settings
- Detection zones
- Recording parameters
- Sensitivity thresholds

### Car Profile (`config/car_profile.yaml`)
- Your car's make, model, year
- Colour details
- Number plate (stored securely)
- Parking position hints

## Detection Pipeline

```
Frame → Car Detected? → Person/Object Near? → Contact Made? → RECORD
           ↓                   ↓                   ↓
          No                  No                  No
           ↓                   ↓                   ↓
        Ignore            Standby              Log only
```

## Troubleshooting

### Hailo not detected
```bash
# Check device
ls -la /dev/hailo*

# Enable PCIe Gen3
sudo raspi-config
# → Advanced Options → PCIe Speed → Yes

# Reboot
sudo reboot
```

### Camera issues
```bash
# Test camera
libcamera-hello

# Check permissions
groups  # Should include 'video'
```

### High false positive rate
Edit `config/config.yaml` and increase:
- `contact_confidence` (default: 0.7)
- `contact_dwell_time` (default: 1.5)

## Legal Considerations

Before deploying:
1. Check your tenancy agreement for camera restrictions
2. Consider local privacy laws
3. Point camera only at your car/parking space
4. Use signage if required in your jurisdiction

This system is designed to minimise privacy impact by only recording genuine interactions with your property.

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please read CLAUDE.md for project context.
