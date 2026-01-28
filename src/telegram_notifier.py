"""
Telegram Notifier Module
========================
Sends notifications to Telegram when contact events are detected.

Features:
- Alert message with snapshot when contact detected
- Video sent when recording ends
- Auto-splits videos >50MB into sequential chunks
- Listens for "me" replies for owner recognition training
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Telegram bot library (async)
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    Bot = None  # Define as None for type hints
    TelegramError = Exception  # Fallback for except clauses
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class TelegramNotifier:
    """
    Handles Telegram notifications for the car monitor.

    Sends alerts when contact is detected and videos when recording ends.
    Automatically splits large videos to stay under Telegram's 50MB limit.
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        send_on_start: bool = True,
        send_video_on_end: bool = True,
        max_chunk_size_mb: int = 50
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.send_on_start = send_on_start
        self.send_video_on_end = send_video_on_end
        self.max_chunk_size_mb = max_chunk_size_mb

        self._bot = None  # Will be Bot instance when initialized
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Listener for "me" replies
        self._listener_thread: Optional[threading.Thread] = None
        self._listener_running = False
        self._last_update_id = 0
        self._owner_callback: Optional[Callable[[int], None]] = None

        if not TELEGRAM_AVAILABLE:
            self.enabled = False
            logger.error("Telegram notifications disabled: python-telegram-bot not installed")
        elif not bot_token or not chat_id:
            self.enabled = False
            logger.warning("Telegram notifications disabled: missing bot_token or chat_id")

    def _get_bot(self):
        """Get or create the Telegram bot instance."""
        if not TELEGRAM_AVAILABLE or not self.enabled:
            return None

        if self._bot is None:
            self._bot = Bot(token=self.bot_token)

        return self._bot

    def _run_async(self, coro):
        """Run an async coroutine from sync code."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop in a thread if one is already running
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(coro)

    def send_alert(
        self,
        message: str,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None
    ) -> bool:
        """
        Send an alert notification to Telegram.

        Args:
            message: Alert text message
            image_path: Optional path to image file to attach
            image_data: Optional raw image bytes (JPEG/PNG)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled or not self.send_on_start:
            return False

        bot = self._get_bot()
        if bot is None:
            return False

        async def _send():
            try:
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as photo:
                        await bot.send_photo(
                            chat_id=self.chat_id,
                            photo=photo,
                            caption=message
                        )
                elif image_data:
                    await bot.send_photo(
                        chat_id=self.chat_id,
                        photo=image_data,
                        caption=message
                    )
                else:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=message
                    )
                return True
            except TelegramError as e:
                logger.error(f"Failed to send Telegram alert: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error sending Telegram alert: {e}")
                return False

        try:
            result = self._run_async(_send())
            if result:
                logger.info(f"Telegram alert sent: {message[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Failed to run async send: {e}")
            return False

    def send_video(
        self,
        video_path: str,
        caption: Optional[str] = None,
        cleanup_chunks: bool = True
    ) -> bool:
        """
        Send a video to Telegram, splitting if necessary.

        Args:
            video_path: Path to the video file
            caption: Optional caption for the video
            cleanup_chunks: Whether to delete chunk files after sending

        Returns:
            True if all parts sent successfully, False otherwise
        """
        if not self.enabled or not self.send_video_on_end:
            return False

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False

        bot = self._get_bot()
        if bot is None:
            return False

        # Check file size and split if needed
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        if file_size_mb <= self.max_chunk_size_mb:
            # Send single video
            chunks = [video_path]
            created_chunks = False
        else:
            # Split into chunks
            logger.info(f"Video {file_size_mb:.1f}MB exceeds limit, splitting...")
            chunks = self._split_video(video_path)
            created_chunks = True

            if not chunks:
                logger.error("Failed to split video")
                return False

        total_parts = len(chunks)
        success = True

        async def _send_all():
            nonlocal success
            for i, chunk_path in enumerate(chunks, 1):
                part_caption = None
                if total_parts > 1:
                    part_caption = f"Part {i}/{total_parts}"
                    if i == 1 and caption:
                        part_caption = f"{caption}\n\n{part_caption}"
                elif caption:
                    part_caption = caption

                try:
                    with open(chunk_path, 'rb') as video:
                        await bot.send_video(
                            chat_id=self.chat_id,
                            video=video,
                            caption=part_caption,
                            supports_streaming=True
                        )
                    logger.info(f"Sent video part {i}/{total_parts}")
                except TelegramError as e:
                    logger.error(f"Failed to send video part {i}/{total_parts}: {e}")
                    success = False
                except Exception as e:
                    logger.error(f"Unexpected error sending video: {e}")
                    success = False

        try:
            self._run_async(_send_all())
        except Exception as e:
            logger.error(f"Failed to send video: {e}")
            success = False

        # Cleanup chunk files
        if created_chunks and cleanup_chunks:
            for chunk_path in chunks:
                try:
                    if chunk_path != video_path and os.path.exists(chunk_path):
                        os.remove(chunk_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup chunk {chunk_path}: {e}")

        # Send summary if multiple parts
        if total_parts > 1 and success:
            self._run_async(bot.send_message(
                chat_id=self.chat_id,
                text=f"Recording complete: {total_parts} parts sent"
            ))

        return success

    def _split_video(self, video_path: str) -> List[str]:
        """
        Split a video into chunks under the size limit.

        Uses ffmpeg to split the video into segments.

        Args:
            video_path: Path to the video file

        Returns:
            List of paths to chunk files
        """
        # Get video duration
        duration = self._get_video_duration(video_path)
        if duration is None:
            logger.error("Could not determine video duration")
            return []

        # Calculate segment duration based on file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        num_segments = int(file_size_mb / self.max_chunk_size_mb) + 1
        segment_duration = duration / num_segments

        # Ensure segments are at least 5 seconds
        segment_duration = max(segment_duration, 5.0)

        # Create output directory
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Use temp directory for chunks
        chunk_dir = tempfile.mkdtemp(prefix="telegram_chunks_")
        output_pattern = os.path.join(chunk_dir, f"{video_name}_part%03d.mp4")

        try:
            # Run ffmpeg to split video
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-c', 'copy',  # Copy without re-encoding (fast)
                '-f', 'segment',
                '-segment_time', str(int(segment_duration)),
                '-reset_timestamps', '1',
                '-y',  # Overwrite
                output_pattern
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg split failed: {result.stderr}")
                return []

            # Collect chunk files
            chunks = sorted(Path(chunk_dir).glob(f"{video_name}_part*.mp4"))
            chunk_paths = [str(c) for c in chunks]

            logger.info(f"Split video into {len(chunk_paths)} chunks")
            return chunk_paths

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg split timed out")
            return []
        except Exception as e:
            logger.error(f"Failed to split video: {e}")
            return []

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration in seconds using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return float(result.stdout.strip())
            return None

        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return None

    def send_recording_start(
        self,
        contact_type: str,
        confidence: float,
        frame_data: Optional[bytes] = None,
        car_id_reasons: Optional[List[str]] = None
    ) -> bool:
        """
        Send notification when recording starts.

        Args:
            contact_type: Type of contact detected (e.g., "HAND_TOUCH")
            confidence: Detection confidence (0-1)
            frame_data: Optional JPEG image data of the frame
            car_id_reasons: How the car was identified (e.g. plate_match, colour_match)

        Returns:
            True if sent successfully
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format car identification info
        id_info = "Unknown"
        if car_id_reasons:
            reason_names = [r.split(':')[0] for r in car_id_reasons]
            id_info = ', '.join(reason_names)

        message = (
            f"Contact detected on your car!\n\n"
            f"Type: {contact_type}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Car ID: {id_info}\n"
            f"Time: {timestamp}\n\n"
            f"Recording started...\n\n"
            f"Reply 'null' if this is wrong (teaches the system)."
        )

        return self.send_alert(message, image_data=frame_data)

    def send_recording_end(
        self,
        video_path: str,
        duration: float,
        contact_type: str
    ) -> bool:
        """
        Send notification when recording ends with video.

        Args:
            video_path: Path to the recorded video
            duration: Recording duration in seconds
            contact_type: Type of contact that triggered recording

        Returns:
            True if sent successfully
        """
        caption = (
            f"Recording complete\n"
            f"Duration: {duration:.1f}s\n"
            f"Event: {contact_type}"
        )

        return self.send_video(video_path, caption=caption)

    def start_listener(self, owner_callback: Callable[[int], None]):
        """
        Start background thread listening for "me" replies.

        Args:
            owner_callback: Function to call when "me" is received.
                           Receives event_id (or 0 for latest event).
        """
        if not self.enabled:
            return

        self._owner_callback = owner_callback
        self._listener_running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="TelegramListener"
        )
        self._listener_thread.start()
        logger.info("Telegram reply listener started")

    def stop_listener(self):
        """Stop the reply listener thread."""
        self._listener_running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=5)
        logger.info("Telegram reply listener stopped")

    def _listener_loop(self):
        """Background loop polling for messages."""
        logger.debug("Listener loop started")

        # Create a dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._listener_running:
            try:
                self._check_for_replies_sync(loop)
            except Exception as e:
                logger.error(f"Listener error: {e}")

            # Poll every 5 seconds
            time.sleep(5)

        loop.close()

    def _check_for_replies_sync(self, loop):
        """Check for new messages using provided event loop."""
        bot = self._get_bot()
        if bot is None:
            return

        async def _get_updates():
            try:
                updates = await bot.get_updates(
                    offset=self._last_update_id + 1,
                    timeout=10,
                    allowed_updates=['message']
                )
                return updates
            except Exception as e:
                logger.warning(f"Failed to get updates: {e}")
                return []

        try:
            updates = loop.run_until_complete(_get_updates())

            for update in updates:
                self._last_update_id = update.update_id

                if not update.message:
                    continue

                # Check if from our chat
                if str(update.message.chat_id) != str(self.chat_id):
                    continue

                text = (update.message.text or "").strip().lower()

                # Check for "me" or variations (owner confirmation)
                if text in ['me', 'mine', 'owner', "that's me", "thats me", "it's me", "its me"]:
                    logger.info("Received owner confirmation reply")
                    self._handle_owner_reply(update.message, loop)

                # Check for "null" or variations (false positive)
                elif text in ['null', 'false', 'no', 'not me', 'notme', 'wrong', 'fp',
                              'false positive', 'nope', 'not mine', 'other car',
                              'neighbour', 'neighbor', 'next car', 'wrong car',
                              'not my car', 'different car', 'ignore']:
                    logger.info("Received false positive reply")
                    self._handle_false_positive_reply(update.message, loop)

        except Exception as e:
            logger.error(f"Error checking replies: {e}")

    def _check_for_replies(self):
        """Legacy method - use _check_for_replies_sync instead."""
        # Keep for backwards compatibility but log warning
        logger.warning("_check_for_replies called without event loop - replies may not work")

    def _handle_owner_reply(self, message, loop=None):
        """Handle a 'me' reply from the owner."""
        # Try to get event_id from replied-to message
        event_id = 0

        if message.reply_to_message and message.reply_to_message.caption:
            # Look for event ID in the original message
            caption = message.reply_to_message.caption
            # Could parse event ID from caption if we include it
            # For now, just use 0 to indicate "latest event"

        # Send confirmation
        async def _reply():
            bot = self._get_bot()
            if bot:
                try:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text="Got it! Learning your appearance..."
                    )
                except Exception as e:
                    logger.error(f"Failed to send confirmation: {e}")

        if loop:
            loop.run_until_complete(_reply())
        else:
            self._run_async(_reply())

        # Trigger callback
        if self._owner_callback:
            try:
                self._owner_callback(event_id)
            except Exception as e:
                logger.error(f"Owner callback failed: {e}")

    def _handle_false_positive_reply(self, message, loop=None):
        """Handle a 'null/false' reply for false positive feedback."""
        event_id = 0

        if message.reply_to_message and message.reply_to_message.caption:
            caption = message.reply_to_message.caption
            # Could parse event ID from caption

        # Send confirmation
        async def _reply():
            bot = self._get_bot()
            if bot:
                try:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=(
                            "False positive recorded. "
                            "The detection zone has been logged and will be "
                            "suppressed in future. The more you report, "
                            "the better the filtering becomes."
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to send FP confirmation: {e}")

        if loop:
            loop.run_until_complete(_reply())
        else:
            self._run_async(_reply())

        # Trigger false positive callback if set
        if hasattr(self, '_false_positive_callback') and self._false_positive_callback:
            try:
                self._false_positive_callback(event_id)
            except Exception as e:
                logger.error(f"False positive callback failed: {e}")

    def set_false_positive_callback(self, callback):
        """Set callback for false positive reports."""
        self._false_positive_callback = callback


def create_notifier_from_config(config: dict) -> Optional[TelegramNotifier]:
    """
    Create a TelegramNotifier from configuration dictionary.

    Args:
        config: Configuration dict with alerts.telegram section

    Returns:
        TelegramNotifier instance or None if disabled/invalid
    """
    alerts_config = config.get('alerts', {})

    if not alerts_config.get('enabled', False):
        logger.info("Alerts disabled in config")
        return None

    telegram_config = alerts_config.get('telegram', {})

    if not telegram_config.get('enabled', False):
        logger.info("Telegram alerts disabled in config")
        return None

    bot_token = telegram_config.get('bot_token', '')
    chat_id = telegram_config.get('chat_id', '')

    if not bot_token or bot_token == 'YOUR_BOT_TOKEN':
        logger.warning("Telegram bot_token not configured")
        return None

    if not chat_id or chat_id == 'YOUR_CHAT_ID':
        logger.warning("Telegram chat_id not configured")
        return None

    return TelegramNotifier(
        bot_token=bot_token,
        chat_id=str(chat_id),
        enabled=True,
        send_on_start=telegram_config.get('send_on_start', True),
        send_video_on_end=telegram_config.get('send_video_on_end', True),
        max_chunk_size_mb=telegram_config.get('max_chunk_size_mb', 50)
    )
