#!/bin/bash
# Pi Car Monitor - Installation Script
# =====================================
# Run this on your Raspberry Pi 5 with Hailo AI HAT

set -e  # Exit on error

echo "=========================================="
echo "Pi Car Monitor - Installation"
echo "=========================================="
echo ""

# Check we're on a Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Warning: This doesn't appear to be a Raspberry Pi."
    echo "Some components may not work correctly."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Pi 5
if grep -q "Raspberry Pi 5" /proc/cpuinfo 2>/dev/null; then
    echo "✓ Detected Raspberry Pi 5"
else
    echo "⚠ This is optimised for Pi 5. Other models may have reduced performance."
fi

# Update system
echo ""
echo "Step 1: Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    sqlite3 \
    git

# Check for Hailo
echo ""
echo "Step 3: Checking Hailo AI HAT..."
if [ -e /dev/hailo0 ]; then
    echo "✓ Hailo device detected at /dev/hailo0"
    
    # Check Hailo tools
    if command -v hailortcli &> /dev/null; then
        echo "✓ HailoRT tools installed"
        hailortcli fw-control identify
    else
        echo "⚠ HailoRT tools not found. Installing..."
        # Install Hailo software
        # This uses the official Raspberry Pi method
        sudo apt install -y hailo-all
    fi
else
    echo "⚠ Hailo device not detected."
    echo "  Make sure your AI HAT is properly connected."
    echo "  Run 'sudo raspi-config' and enable PCIe Gen3 under Advanced Options."
    echo "  Then reboot and run this script again."
    read -p "Continue without Hailo? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo ""
echo "Step 4: Creating Python virtual environment..."
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo ""
echo "Step 6: Creating directory structure..."
mkdir -p data/recordings
mkdir -p data/reference_images
mkdir -p data/calibration
mkdir -p data/debug
mkdir -p models/custom
mkdir -p logs

# Set permissions
chmod 700 config/
touch logs/events.db

# Download base models
echo ""
echo "Step 7: Downloading base detection models..."
if [ -f models/download_models.sh ]; then
    bash models/download_models.sh
else
    echo "  Model download script not found. You'll need to download models manually."
fi

# Setup systemd service (optional)
echo ""
read -p "Step 8: Install as system service (auto-start on boot)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo cp systemd/car-monitor.service /etc/systemd/system/
    sudo systemctl daemon-reload
    echo "✓ Service installed. Enable with: sudo systemctl enable car-monitor"
fi

# Final instructions
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit your car profile:"
echo "   nano config/car_profile.yaml"
echo ""
echo "2. Run calibration to set up detection zones:"
echo "   source venv/bin/activate"
echo "   python src/main.py --calibrate"
echo ""
echo "3. Test the system:"
echo "   python src/main.py --test"
echo ""
echo "4. Start monitoring:"
echo "   python src/main.py"
echo ""
echo "For help with Claude Code:"
echo "   Install Claude Code: npm install -g @anthropic-ai/claude-code"
echo "   Run: claude"
echo "   Claude will read CLAUDE.md and understand the project context."
echo ""
