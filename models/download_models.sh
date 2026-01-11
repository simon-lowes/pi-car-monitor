#!/bin/bash
# Download Hailo-optimised detection models
# ==========================================
# Downloads pre-compiled HEF models for Hailo-8L
#
# Models needed:
# 1. YOLOv8n - General object detection (cars, people, etc.)
# 2. YOLOv8n-pose - Pose estimation for contact detection
# 3. LPRNet - License plate recognition (optional)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Downloading Hailo Detection Models"
echo "=========================================="
echo ""

# Check for Hailo tools
if ! command -v hailortcli &> /dev/null; then
    echo "Warning: hailortcli not found. Models may not be compatible."
    echo "Install with: sudo apt install hailo-all"
fi

# Detect Hailo device type
HAILO_ARCH="hailo8l"  # Default for AI HAT
if [ -e /dev/hailo0 ]; then
    HAILO_INFO=$(hailortcli fw-control identify 2>/dev/null || true)
    if echo "$HAILO_INFO" | grep -q "HAILO-8"; then
        if echo "$HAILO_INFO" | grep -q "hailo8l"; then
            HAILO_ARCH="hailo8l"
            echo "Detected: Hailo-8L (AI HAT)"
        else
            HAILO_ARCH="hailo8"
            echo "Detected: Hailo-8"
        fi
    fi
else
    echo "No Hailo device detected - downloading hailo8l models (AI HAT default)"
fi

echo ""

# Base URL for Hailo Model Zoo (official releases)
# Note: These URLs may change - check hailo.ai for latest
HAILO_MODEL_ZOO="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled"

# Alternative: Use rpicam-apps models if available
RPICAM_MODELS="/usr/share/hailo-models"

download_model() {
    local name=$1
    local url=$2
    local output=$3

    if [ -f "$output" ]; then
        echo "  [exists] $name"
        return 0
    fi

    echo "  Downloading $name..."
    if curl -fsSL "$url" -o "$output" 2>/dev/null; then
        echo "  [done] $name"
        return 0
    else
        echo "  [failed] $name - will try alternative sources"
        return 1
    fi
}

copy_system_model() {
    local name=$1
    local source=$2
    local output=$3

    if [ -f "$output" ]; then
        echo "  [exists] $name"
        return 0
    fi

    if [ -f "$source" ]; then
        cp "$source" "$output"
        echo "  [copied] $name from system"
        return 0
    fi

    return 1
}

echo "Downloading object detection models..."
echo ""

# Try to copy from system rpicam-apps installation first
if [ -d "$RPICAM_MODELS" ]; then
    echo "Found system Hailo models at $RPICAM_MODELS"

    # YOLOv8n for object detection
    copy_system_model "yolov8n" \
        "$RPICAM_MODELS/yolov8n_${HAILO_ARCH}.hef" \
        "yolov8n.hef" || true

    # YOLOv8n pose estimation
    copy_system_model "yolov8n_pose" \
        "$RPICAM_MODELS/yolov8n_pose_${HAILO_ARCH}.hef" \
        "yolov8n_pose.hef" || true

    # YOLOv5/v6 alternatives
    copy_system_model "yolov5n_seg" \
        "$RPICAM_MODELS/yolov5n_seg_${HAILO_ARCH}.hef" \
        "yolov5n_seg.hef" || true
fi

# Download from Hailo Model Zoo if not found locally
echo ""
echo "Checking for missing models..."

# YOLOv8n - Object detection (primary model)
if [ ! -f "yolov8n.hef" ]; then
    echo "  Attempting to download yolov8n from Hailo Model Zoo..."
    download_model "yolov8n" \
        "${HAILO_MODEL_ZOO}/${HAILO_ARCH}/yolov8n.hef" \
        "yolov8n.hef" || true
fi

# YOLOv8n Pose - Pose estimation for contact detection
if [ ! -f "yolov8n_pose.hef" ]; then
    echo "  Attempting to download yolov8n_pose from Hailo Model Zoo..."
    download_model "yolov8n_pose" \
        "${HAILO_MODEL_ZOO}/${HAILO_ARCH}/yolov8n_pose.hef" \
        "yolov8n_pose.hef" || true
fi

echo ""
echo "=========================================="
echo "Model Status"
echo "=========================================="

check_model() {
    local name=$1
    local file=$2
    local required=$3

    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "[OK]       $name ($size)"
    elif [ "$required" = "required" ]; then
        echo "[MISSING]  $name (REQUIRED)"
    else
        echo "[MISSING]  $name (optional)"
    fi
}

check_model "YOLOv8n Detection" "yolov8n.hef" "required"
check_model "YOLOv8n Pose" "yolov8n_pose.hef" "required"
check_model "YOLOv5n Segmentation" "yolov5n_seg.hef" "optional"

echo ""

# Check custom models directory
echo "Custom models directory: custom/"
if [ -d "custom" ]; then
    custom_count=$(ls -1 custom/*.hef 2>/dev/null | wc -l)
    echo "  Found $custom_count custom model(s)"
else
    mkdir -p custom
    echo "  Created (empty) - add fine-tuned models here"
fi

echo ""

# Final instructions
if [ ! -f "yolov8n.hef" ] || [ ! -f "yolov8n_pose.hef" ]; then
    echo "=========================================="
    echo "MANUAL DOWNLOAD REQUIRED"
    echo "=========================================="
    echo ""
    echo "Some models couldn't be downloaded automatically."
    echo ""
    echo "Option 1: Install from apt (recommended for RPi)"
    echo "  sudo apt install hailo-all"
    echo "  Models will be at /usr/share/hailo-models/"
    echo ""
    echo "Option 2: Download from Hailo Model Zoo"
    echo "  Visit: https://github.com/hailo-ai/hailo_model_zoo"
    echo "  Download HEF files for $HAILO_ARCH"
    echo "  Place in: $SCRIPT_DIR/"
    echo ""
    echo "Option 3: Compile from ONNX"
    echo "  Use Hailo Dataflow Compiler to compile"
    echo "  YOLOv8n ONNX models for $HAILO_ARCH"
    echo ""
else
    echo "All required models present!"
    echo ""
    echo "Next step: Run calibration"
    echo "  cd $(dirname $SCRIPT_DIR)"
    echo "  source venv/bin/activate"
    echo "  python src/main.py --calibrate"
fi
