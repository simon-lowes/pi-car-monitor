"""
Camera Module
=============
Handles camera capture using picamera2 on Raspberry Pi 5.

Provides:
- Efficient frame capture for real-time processing
- Configurable resolution and framerate
- Integration with Hailo pipeline via GStreamer (future)
"""

import logging
import threading
import time
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Camera interface for Pi Camera Module.

    Uses picamera2 for capture on Raspberry Pi 5.
    Falls back to OpenCV VideoCapture for USB cameras or testing.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 15,
        use_picamera: bool = True
    ):
        self.device = device
        self.resolution = resolution
        self.framerate = framerate
        self.use_picamera = use_picamera

        self._camera = None
        self._running = False
        self._lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_timestamp: float = 0

        # Frame statistics
        self.frames_captured = 0
        self.capture_errors = 0

    def start(self) -> bool:
        """Start the camera."""
        if self._running:
            logger.warning("Camera already running")
            return True

        if self.use_picamera:
            success = self._start_picamera()
        else:
            success = self._start_opencv()

        if success:
            self._running = True
            logger.info(f"Camera started: {self.resolution[0]}x{self.resolution[1]} @ {self.framerate}fps")

        return success

    def _start_picamera(self) -> bool:
        """Initialize picamera2."""
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()

            # Configure for video capture
            config = self._camera.create_video_configuration(
                main={"size": self.resolution, "format": "BGR888"},
                controls={"FrameRate": self.framerate}
            )
            self._camera.configure(config)
            self._camera.start()

            # Wait for camera to stabilize
            time.sleep(0.5)

            logger.info("picamera2 initialized")
            return True

        except ImportError:
            logger.warning("picamera2 not available, falling back to OpenCV")
            return self._start_opencv()
        except Exception as e:
            logger.error(f"Failed to initialize picamera2: {e}")
            return self._start_opencv()

    def _start_opencv(self) -> bool:
        """Initialize OpenCV VideoCapture as fallback."""
        try:
            import cv2

            # Try device path first, then index
            if self.device.startswith("/dev/"):
                cap = cv2.VideoCapture(self.device)
            else:
                cap = cv2.VideoCapture(int(self.device) if self.device.isdigit() else 0)

            if not cap.isOpened():
                logger.error(f"Failed to open camera: {self.device}")
                return False

            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.framerate)

            self._camera = cap
            self.use_picamera = False

            logger.info("OpenCV VideoCapture initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OpenCV camera: {e}")
            return False

    def stop(self):
        """Stop the camera."""
        if not self._running:
            return

        with self._lock:
            self._running = False

            if self._camera is not None:
                if self.use_picamera:
                    try:
                        self._camera.stop()
                        self._camera.close()
                    except Exception as e:
                        logger.warning(f"Error stopping picamera2: {e}")
                else:
                    try:
                        self._camera.release()
                    except Exception as e:
                        logger.warning(f"Error releasing OpenCV camera: {e}")

                self._camera = None

        logger.info(f"Camera stopped. Frames captured: {self.frames_captured}")

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.

        Returns:
            BGR numpy array or None if capture failed
        """
        if not self._running or self._camera is None:
            return None

        with self._lock:
            try:
                if self.use_picamera:
                    frame = self._camera.capture_array()
                else:
                    ret, frame = self._camera.read()
                    if not ret:
                        self.capture_errors += 1
                        return None

                self._last_frame = frame
                self._last_timestamp = time.time()
                self.frames_captured += 1

                return frame

            except Exception as e:
                self.capture_errors += 1
                logger.error(f"Frame capture error: {e}")
                return None

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """Get the most recently captured frame."""
        return self._last_frame

    @property
    def last_timestamp(self) -> float:
        """Get timestamp of last captured frame."""
        return self._last_timestamp

    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get camera statistics."""
        return {
            "running": self._running,
            "frames_captured": self.frames_captured,
            "capture_errors": self.capture_errors,
            "resolution": self.resolution,
            "framerate": self.framerate,
            "backend": "picamera2" if self.use_picamera else "opencv"
        }


class ThreadedCamera(Camera):
    """
    Camera with background capture thread for non-blocking operation.

    Continuously captures frames in background, always has latest
    frame available without blocking the main processing loop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capture_thread: Optional[threading.Thread] = None
        self._frame_ready = threading.Event()

    def start(self) -> bool:
        """Start camera with background capture thread."""
        if not super().start():
            return False

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._capture_thread.start()

        # Wait for first frame
        self._frame_ready.wait(timeout=2.0)

        return True

    def stop(self):
        """Stop camera and capture thread."""
        self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        super().stop()

    def _capture_loop(self):
        """Background capture loop."""
        interval = 1.0 / self.framerate

        while self._running:
            start = time.time()

            frame = super().capture()
            if frame is not None:
                self._frame_ready.set()

            # Maintain framerate
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get the latest frame (non-blocking).

        Args:
            timeout: Max time to wait for a frame

        Returns:
            Latest frame or None
        """
        if self._frame_ready.wait(timeout=timeout):
            return self._last_frame
        return None
