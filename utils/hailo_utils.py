"""
Hailo Utilities
================
Helper functions and classes for Hailo AI accelerator inference.

Supports Hailo-8L (AI HAT) with HailoRT.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Hailo runtime
try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo_platform not available - inference will be disabled")


def check_hailo_available() -> bool:
    """Check if Hailo device is available and working."""
    # Check device exists
    if not Path("/dev/hailo0").exists():
        logger.warning("Hailo device /dev/hailo0 not found")
        return False

    # Try to identify the device
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.debug(f"Hailo device info: {result.stdout}")
            return True
        else:
            logger.warning(f"Hailo identify failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning("hailortcli not found - is HailoRT installed?")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Hailo identify timed out")
        return False
    except Exception as e:
        logger.warning(f"Hailo check failed: {e}")
        return False


def get_hailo_info() -> Optional[dict]:
    """Get detailed information about the Hailo device."""
    if not check_hailo_available():
        return None

    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True,
            text=True,
            timeout=10
        )

        info = {}
        for line in result.stdout.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()

        return info
    except Exception as e:
        logger.error(f"Failed to get Hailo info: {e}")
        return None


class HailoDevice:
    """
    Wrapper for Hailo device operations.

    Handles model loading, inference, and resource management.
    """

    def __init__(self, device_path: str = "/dev/hailo0"):
        self.device_path = device_path
        self.models: Dict[str, 'HailoModel'] = {}
        self._vdevice = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the Hailo device."""
        if not HAILO_AVAILABLE:
            logger.error("Hailo platform not available")
            return False

        if not check_hailo_available():
            return False

        try:
            # Create virtual device
            self._vdevice = VDevice()
            self._initialized = True
            logger.info("Hailo device initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Hailo device: {e}")
            return False

    def load_model(self, model_path: str, name: str = None) -> bool:
        """
        Load a HEF model onto the device.

        Args:
            model_path: Path to .hef file
            name: Name to reference this model

        Returns:
            True if successful
        """
        if not self._initialized:
            logger.error("Device not initialized")
            return False

        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        name = name or model_path.stem

        try:
            model = HailoModel(str(model_path), self._vdevice)
            if model.load():
                self.models[name] = model
                logger.info(f"Loaded model: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            return False

    def run_inference(self, model_name: str, input_data: np.ndarray) -> Optional[Dict]:
        """
        Run inference on loaded model.

        Args:
            model_name: Name of loaded model
            input_data: Input image as numpy array (BGR, HWC format)

        Returns:
            Dictionary of output tensors or None on error
        """
        if model_name not in self.models:
            logger.error(f"Model not loaded: {model_name}")
            return None

        return self.models[model_name].infer(input_data)

    def cleanup(self):
        """Clean up Hailo resources."""
        for name, model in self.models.items():
            try:
                model.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up model {name}: {e}")

        self.models.clear()
        self._vdevice = None
        self._initialized = False
        logger.info("Hailo device cleaned up")


class HailoModel:
    """
    Wrapper for a single Hailo model.

    Handles preprocessing, inference, and postprocessing.
    """

    def __init__(self, hef_path: str, vdevice):
        self.hef_path = hef_path
        self._vdevice = vdevice
        self._hef = None
        self._network_group = None
        self._input_vstreams = None
        self._output_vstreams = None
        self._input_shape = None
        self._loaded = False

    def load(self) -> bool:
        """Load the HEF file and configure streams."""
        if not HAILO_AVAILABLE:
            return False

        try:
            # Load HEF
            self._hef = HEF(self.hef_path)

            # Configure network group
            configure_params = ConfigureParams.create_from_hef(
                self._hef,
                interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(
                self._hef,
                configure_params
            )[0]

            # Create network group params for activation
            self._network_group_params = self._network_group.create_params()

            # Get input/output info
            input_vstream_info = self._hef.get_input_vstream_infos()[0]
            self._input_shape = input_vstream_info.shape

            self._loaded = True
            logger.debug(f"Model loaded: {self.hef_path}")
            logger.debug(f"Input shape: {self._input_shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def infer(self, image: np.ndarray) -> Optional[Dict]:
        """
        Run inference on an image.

        Args:
            image: BGR image as numpy array

        Returns:
            Dictionary mapping output names to tensors
        """
        if not self._loaded:
            logger.error("Model not loaded")
            return None

        try:
            # Debug: log input image info
            logger.debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

            # Preprocess image
            input_data = self._preprocess(image)

            # Debug: log preprocessed data info
            input_name = self._hef.get_input_vstream_infos()[0].name
            logger.debug(f"Preprocessed shape: {input_data.shape}, dtype: {input_data.dtype}, "
                        f"size: {input_data.nbytes} bytes, contiguous: {input_data.flags['C_CONTIGUOUS']}")
            logger.debug(f"Input name: {input_name}, expected shape: {self._input_shape}")

            # Create vstream params - use UINT8 input as model expects
            input_vstream_params = InputVStreamParams.make_from_network_group(
                self._network_group,
                quantized=True,
                format_type=FormatType.UINT8
            )
            output_vstream_params = OutputVStreamParams.make_from_network_group(
                self._network_group,
                quantized=False,
                format_type=FormatType.FLOAT32
            )

            # input_data already has batch dimension from _preprocess()
            logger.debug(f"Input strides: {input_data.strides}")

            # Run inference with network group activation
            with self._network_group.activate(self._network_group_params):
                with InferVStreams(
                    self._network_group,
                    input_vstream_params,
                    output_vstream_params
                ) as infer_pipeline:
                    input_dict = {
                        input_name: input_data
                    }
                    results = infer_pipeline.infer(input_dict)

            return results

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: BGR image (H, W, C)

        Returns:
            Preprocessed tensor (1, H, W, C) as UINT8
        """
        import cv2

        # Get target size from input shape
        # Shape is typically (height, width, channels) for Hailo
        if len(self._input_shape) == 3:
            target_h, target_w = self._input_shape[0], self._input_shape[1]
        elif len(self._input_shape) == 4:
            if self._input_shape[3] == 3:  # NHWC
                target_h, target_w = self._input_shape[1], self._input_shape[2]
            else:  # NCHW
                target_h, target_w = self._input_shape[2], self._input_shape[3]
        else:
            target_h, target_w = 640, 640  # Default YOLOv8 size

        # Resize
        resized = cv2.resize(image, (target_w, target_h))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Keep as UINT8 - Hailo model expects UINT8 input!
        # Hailo InferVStreams requires shape (batch, H, W, C)
        batched = np.expand_dims(rgb, axis=0).astype(np.uint8)

        return batched

    def cleanup(self):
        """Release resources."""
        self._network_group = None
        self._hef = None
        self._loaded = False


def postprocess_yolov8_detections(
    outputs: Dict,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    input_shape: Tuple[int, int] = (640, 640),
    orig_shape: Tuple[int, int] = (1080, 1920)
) -> List[Dict]:
    """
    Postprocess YOLOv8 detection outputs.

    Args:
        outputs: Raw model outputs
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        input_shape: Model input size (H, W)
        orig_shape: Original image size (H, W)

    Returns:
        List of detections with keys: bbox, confidence, class_id, class_name
    """
    # COCO class names for vehicle/person detection
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        5: 'bus', 7: 'truck', 16: 'dog', 17: 'cat'
    }

    detections = []

    try:
        # Get first output tensor
        output_name = list(outputs.keys())[0]
        output = outputs[output_name]

        if output is None:
            return []

        # Handle Hailo NMS output format: list[batch][class_id] = array(N, 5)
        # Each detection array contains [x1, y1, x2, y2, confidence]
        if isinstance(output, list) and len(output) > 0:
            # Get first batch
            batch_output = output[0]

            if isinstance(batch_output, list):
                # NMS format: iterate over classes
                for class_id, class_detections in enumerate(batch_output):
                    if class_id not in COCO_CLASSES:
                        continue

                    if not isinstance(class_detections, np.ndarray):
                        continue

                    if class_detections.size == 0:
                        continue

                    # class_detections shape: (N, 5) = [x1, y1, x2, y2, conf]
                    # Hailo NMS outputs normalized coords (0-1)
                    for det in class_detections:
                        if len(det) < 5:
                            continue

                        x1, y1, x2, y2, conf = det[:5]

                        if conf < conf_threshold:
                            continue

                        # Coords are normalized (0-1), scale directly to original image
                        bbox = (
                            int(x1 * orig_shape[1]),
                            int(y1 * orig_shape[0]),
                            int(x2 * orig_shape[1]),
                            int(y2 * orig_shape[0])
                        )

                        detections.append({
                            'bbox': bbox,
                            'confidence': float(conf),
                            'class_id': class_id,
                            'class': COCO_CLASSES[class_id]
                        })

                # Hailo's NMS model already applies NMS, so just return
                return detections

        # Fallback: handle numpy array output (non-NMS models)
        if hasattr(output, 'size') and output.size == 0:
            return []

        # Squeeze batch dimension if numpy array
        if hasattr(output, 'squeeze'):
            output = np.squeeze(output)

        # Determine output format and parse
        if len(output.shape) == 2:
            if output.shape[1] == 6:
                # Format: (N, 6) where each row is [x1, y1, x2, y2, conf, class]
                for det in output:
                    conf = det[4]
                    if conf < conf_threshold:
                        continue

                    class_id = int(det[5])
                    if class_id not in COCO_CLASSES:
                        continue

                    # Scale bbox to original image size
                    scale_x = orig_shape[1] / input_shape[1]
                    scale_y = orig_shape[0] / input_shape[0]

                    bbox = (
                        int(det[0] * scale_x),
                        int(det[1] * scale_y),
                        int(det[2] * scale_x),
                        int(det[3] * scale_y)
                    )

                    detections.append({
                        'bbox': bbox,
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class': COCO_CLASSES[class_id]
                    })

            elif output.shape[0] == 84 or output.shape[1] == 84:
                # Format: (84, 8400) - need to transpose and decode
                if output.shape[0] == 84:
                    output = output.T  # Now (8400, 84)

                # First 4 values are box coords, rest are class probs
                boxes = output[:, :4]
                scores = output[:, 4:]

                # Get best class for each detection
                class_ids = np.argmax(scores, axis=1)
                confidences = np.max(scores, axis=1)

                # Filter by confidence and relevant classes
                for i in range(len(confidences)):
                    if confidences[i] < conf_threshold:
                        continue

                    class_id = class_ids[i]
                    if class_id not in COCO_CLASSES:
                        continue

                    # Decode box (center format to corner format)
                    cx, cy, w, h = boxes[i]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    # Scale to original image
                    scale_x = orig_shape[1] / input_shape[1]
                    scale_y = orig_shape[0] / input_shape[0]

                    bbox = (
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y)
                    )

                    detections.append({
                        'bbox': bbox,
                        'confidence': float(confidences[i]),
                        'class_id': int(class_id),
                        'class': COCO_CLASSES[class_id]
                    })

        # Apply NMS
        if detections:
            detections = _apply_nms(detections, iou_threshold)

    except Exception as e:
        logger.error(f"Postprocessing error: {e}")

    return detections


def _apply_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Apply non-maximum suppression to detections."""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        detections = [
            det for det in detections
            if _iou(best['bbox'], det['bbox']) < iou_threshold
        ]

    return keep


def _iou(box1: Tuple, box2: Tuple) -> float:
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
