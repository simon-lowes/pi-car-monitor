"""
Recorder - Circular Buffer Video Recording
==========================================
Maintains a rolling buffer of recent frames and can trigger
recording when contact is detected.

Key features:
- Circular buffer keeps last N seconds in memory
- On trigger, saves buffer + continues recording
- Automatic cleanup of old recordings
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Deque
import json

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferedFrame:
    """A frame in the circular buffer."""
    data: np.ndarray
    timestamp: float


class CircularRecorder:
    """
    Manages circular buffer recording for contact events.
    
    The buffer always contains the last N seconds of video.
    When recording is triggered, the buffer contents are saved
    followed by live recording until the event ends.
    """
    
    def __init__(
        self,
        output_dir: str,
        pre_buffer_seconds: float = 10,
        post_buffer_seconds: float = 5,
        max_duration: float = 300,
        format: str = "mp4",
        fps: int = 15
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pre_buffer_seconds = pre_buffer_seconds
        self.post_buffer_seconds = post_buffer_seconds
        self.max_duration = max_duration
        self.format = format
        self.fps = fps
        
        # Calculate buffer size
        self.buffer_size = int(pre_buffer_seconds * fps)
        self.buffer: Deque[BufferedFrame] = deque(maxlen=self.buffer_size)
        
        # Recording state
        self._running = False
        self._is_recording = False
        self._recording_start_time: Optional[float] = None
        self._current_recording_path: Optional[Path] = None
        self._recording_metadata: dict = {}
        
        # Video writer
        self._writer = None
        self._writer_lock = threading.Lock()
        
        # Statistics
        self.recordings_count = 0
        self.total_recorded_seconds = 0
    
    def start(self):
        """Start the recorder (enables buffer)."""
        self._running = True
        logger.info(f"Recorder started (buffer: {self.pre_buffer_seconds}s)")
    
    def stop(self):
        """Stop the recorder."""
        if self._is_recording:
            self.stop_recording()
        self._running = False
        self.buffer.clear()
        logger.info("Recorder stopped")
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add a frame to the circular buffer."""
        if not self._running:
            return
        
        self.buffer.append(BufferedFrame(data=frame, timestamp=timestamp))
        
        # If recording, write frame to file
        if self._is_recording:
            self._write_frame(frame, timestamp)
    
    def start_recording(self, reason: str = "contact", metadata: dict = None):
        """
        Start recording: save buffer + continue live recording.
        
        Args:
            reason: Why recording was triggered
            metadata: Additional metadata about the event
        """
        if self._is_recording:
            logger.debug("Already recording")
            return
        
        with self._writer_lock:
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp_str}_{reason}.{self.format}"
            self._current_recording_path = self.output_dir / filename
            
            # Store metadata
            self._recording_metadata = {
                "start_time": datetime.now().isoformat(),
                "reason": reason,
                "pre_buffer_seconds": self.pre_buffer_seconds,
                **(metadata or {})
            }
            
            # Initialize video writer
            if len(self.buffer) > 0:
                frame_shape = self.buffer[0].data.shape
                self._init_writer(frame_shape)
                
                # Write buffered frames
                for buffered_frame in self.buffer:
                    self._write_frame_internal(buffered_frame.data)
                
                logger.info(f"Wrote {len(self.buffer)} buffered frames")
            
            self._is_recording = True
            self._recording_start_time = time.time()
            self.recordings_count += 1
            
            logger.info(f"Recording started: {self._current_recording_path}")
    
    def stop_recording(self):
        """Stop the current recording."""
        if not self._is_recording:
            return
        
        with self._writer_lock:
            # Finalize video
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            
            # Calculate duration
            if self._recording_start_time:
                duration = time.time() - self._recording_start_time
                self.total_recorded_seconds += duration
                self._recording_metadata["duration_seconds"] = duration
            
            # Save metadata
            if self._current_recording_path:
                metadata_path = self._current_recording_path.with_suffix(".json")
                with open(metadata_path, "w") as f:
                    json.dump(self._recording_metadata, f, indent=2)
            
            self._is_recording = False
            self._recording_start_time = None
            
            logger.info(f"Recording saved: {self._current_recording_path}")
            
            self._current_recording_path = None
            self._recording_metadata = {}
    
    def _init_writer(self, frame_shape):
        """Initialize the video writer."""
        try:
            import cv2
            
            height, width = frame_shape[:2]
            
            # Choose codec based on format
            if self.format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self._writer = cv2.VideoWriter(
                str(self._current_recording_path),
                fourcc,
                self.fps,
                (width, height)
            )
            
            if not self._writer.isOpened():
                logger.error("Failed to open video writer")
                self._writer = None
                
        except ImportError:
            logger.error("OpenCV not available for video writing")
            self._writer = None
    
    def _write_frame(self, frame: np.ndarray, timestamp: float):
        """Write a frame during live recording."""
        with self._writer_lock:
            if not self._is_recording:
                return
            
            # Check max duration
            if self._recording_start_time:
                duration = timestamp - self._recording_start_time
                if duration > self.max_duration:
                    logger.warning(f"Max recording duration reached ({self.max_duration}s)")
                    self.stop_recording()
                    return
            
            self._write_frame_internal(frame)
    
    def _write_frame_internal(self, frame: np.ndarray):
        """Internal frame writing (no locks)."""
        if self._writer is not None and self._writer.isOpened():
            self._writer.write(frame)
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    @property
    def current_recording_duration(self) -> float:
        """Get current recording duration in seconds."""
        if self._is_recording and self._recording_start_time:
            return time.time() - self._recording_start_time
        return 0.0
    
    def get_stats(self) -> dict:
        """Get recorder statistics."""
        return {
            "is_recording": self._is_recording,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "recordings_count": self.recordings_count,
            "total_recorded_seconds": self.total_recorded_seconds,
            "current_duration": self.current_recording_duration
        }
    
    def cleanup_old_recordings(self, max_age_days: int = 30):
        """Delete recordings older than max_age_days."""
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)
        deleted = 0
        
        for recording in self.output_dir.glob(f"*.{self.format}"):
            if recording.stat().st_mtime < cutoff:
                # Delete video and metadata
                recording.unlink()
                metadata = recording.with_suffix(".json")
                if metadata.exists():
                    metadata.unlink()
                deleted += 1
                logger.info(f"Deleted old recording: {recording.name}")
        
        if deleted:
            logger.info(f"Cleaned up {deleted} old recordings")
        
        return deleted
