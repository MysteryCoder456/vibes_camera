"""Alert management for security camera application.

Provides a protocol-based interface for sending alerts when people are detected.
Uses a background thread with a persistent Discord connection for non-blocking
alert delivery.
"""

from __future__ import annotations

import asyncio
import datetime
import io
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Protocol

import cv2
from dotenv import load_dotenv

if TYPE_CHECKING:
    import discord

# Get logger from opencv_project
logger = logging.getLogger("opencv_project")


class AlertManager(Protocol):
    """Protocol defining the interface for alert managers.

    Implementations must provide methods for checking alert conditions
    and sending alerts via their respective providers. The send_alert
    method should be non-blocking.
    """

    def should_alert(self, current_count: int) -> bool:
        """Check if alert should be sent based on current detection count.

        Args:
            current_count: Current number of people detected

        Returns:
            bool: True if alert should be sent
        """
        ...

    def send_alert(self, frame, num_people: int, duration: float = 0.0) -> None:
        """Queue alert for async sending (non-blocking).

        Args:
            frame: OpenCV frame with detections drawn
            num_people: Number of people detected
            duration: How long the face was present before alert (seconds)
        """
        ...

    def update_count(self, current_count: int) -> None:
        """Update the previous detection count.

        Args:
            current_count: Current number of people detected
        """
        ...

    def shutdown(self) -> None:
        """Clean up resources and stop background workers."""
        ...


def load_alert_config():
    """Load Discord bot credentials from environment.

    Returns:
        dict with credentials, or None if not configured
    """
    load_dotenv()

    required_vars = [
        "DISCORD_BOT_TOKEN",
        "DISCORD_USER_ID",
    ]

    config = {}
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            config[var] = value

    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")
        logger.warning("Please configure .env file (see .env.example)")
        return None

    return config


class DiscordAlertManager:
    """Alert manager that sends DMs via a persistent Discord bot connection.

    The Discord bot connects once on initialization and stays connected for
    the lifetime of the application. Alerts are queued and sent asynchronously
    in a background thread. Handles disconnection/reconnection gracefully.

    Implements retry logic with exponential backoff for failed sends.
    """

    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0  # seconds
    QUEUE_MAX_SIZE = 5
    INIT_TIMEOUT = 30.0  # seconds to wait for initial connection
    RECONNECT_TIMEOUT = 60.0  # seconds to wait for reconnection

    def __init__(self, config, cooldown_seconds=300):
        """Initialize DiscordAlertManager.

        Blocks until the Discord bot connects successfully.

        Args:
            config: Dict with DISCORD_BOT_TOKEN and DISCORD_USER_ID
            cooldown_seconds: Minimum seconds between alerts

        Raises:
            RuntimeError: If bot fails to connect within timeout
            Exception: If connection fails for other reasons (bad token, etc.)
        """
        self.config = config
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = 0
        self.previous_count = 0

        # Queue for non-blocking alert sending
        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAX_SIZE)
        self._shutdown_event = threading.Event()

        # Discord state (set in worker thread)
        self._client: discord.Client | None = None
        self._dm_channel: discord.DMChannel | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Connection state events
        self._ready_event = threading.Event()  # Set when bot first connects
        self._connected_event: asyncio.Event | None = None  # For reconnection tracking
        self._init_error: Exception | None = None  # Stores init failure reason

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="DiscordAlertWorker",
        )
        self._worker_thread.start()

        # Block until connected or failure
        if not self._ready_event.wait(timeout=self.INIT_TIMEOUT):
            self._shutdown_event.set()
            raise RuntimeError(
                f"Discord bot failed to connect within {self.INIT_TIMEOUT}s timeout"
            )

        if self._init_error:
            self._shutdown_event.set()
            raise self._init_error

    def _worker(self):
        """Background worker with persistent Discord connection."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_bot())
        except Exception as e:
            # Store error for main thread to raise
            if self._init_error is None:
                self._init_error = e
            self._ready_event.set()  # Unblock constructor
        finally:
            self._loop.close()

    async def _run_bot(self):
        """Main async function that manages the Discord bot and processes alerts."""
        import discord

        intents = discord.Intents.default()
        self._client = discord.Client(intents=intents)
        self._connected_event = asyncio.Event()

        @self._client.event
        async def on_ready():
            """Called when bot connects/reconnects to Discord."""
            try:
                user_id = int(self.config["DISCORD_USER_ID"])
                user = await self._client.fetch_user(user_id)
                self._dm_channel = await user.create_dm()
                self._connected_event.set()
                logger.info(
                    f"Discord bot connected, DM channel ready for user {user_id}"
                )
                self._ready_event.set()  # Unblock constructor (first connect only)
            except Exception as e:
                if self._init_error is None:
                    self._init_error = e
                self._ready_event.set()

        @self._client.event
        async def on_disconnect():
            """Called when bot disconnects from Discord."""
            logger.warning("Discord bot disconnected, waiting for reconnection...")
            if self._connected_event:
                self._connected_event.clear()

        @self._client.event
        async def on_resumed():
            """Called when bot reconnects after a disconnect."""
            logger.info("Discord bot reconnected")
            if self._connected_event:
                self._connected_event.set()

        # Start the Discord client as a background task
        bot_task = asyncio.create_task(
            self._client.start(self.config["DISCORD_BOT_TOKEN"])
        )

        # Process alert queue
        try:
            await self._process_queue()
        finally:
            if not self._client.is_closed():
                await self._client.close()
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass

    async def _process_queue(self):
        """Process the alert queue, handling disconnections."""
        while not self._shutdown_event.is_set():
            # Check for items in queue
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            if item is None:  # Shutdown signal
                break

            frame, num_people, timestamp, duration = item

            # Wait for connection if disconnected
            if self._connected_event and not self._connected_event.is_set():
                logger.warning(
                    "Waiting for Discord reconnection before sending alert..."
                )
                try:
                    await asyncio.wait_for(
                        self._connected_event.wait(),
                        timeout=self.RECONNECT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Reconnection timeout, discarding alert")
                    self._queue.task_done()
                    continue

            # Send the alert
            await self._send_with_retry(frame, num_people, timestamp, duration)
            self._queue.task_done()

    async def _send_with_retry(
        self, frame, num_people: int, timestamp: str, duration: float
    ):
        """Send Discord DM with retry logic and exponential backoff.

        Args:
            frame: OpenCV frame (already copied)
            num_people: Number of people detected
            timestamp: Timestamp string for the alert
            duration: How long the face was present before alert (seconds)
        """
        import discord

        last_error = None

        for attempt in range(self.MAX_RETRIES):
            # Check if still connected before each attempt
            if self._connected_event and not self._connected_event.is_set():
                logger.warning("Disconnected during retry, waiting for reconnection...")
                try:
                    await asyncio.wait_for(
                        self._connected_event.wait(),
                        timeout=self.RECONNECT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Reconnection timeout during retry, discarding alert"
                    )
                    return

            try:
                await self._send_dm(frame, num_people, timestamp, duration)
                return  # Success
            except (discord.errors.Forbidden, discord.errors.NotFound) as e:
                # Don't retry on permission/not found errors
                logger.error(f"Discord alert failed (not retrying): {e}")
                return
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2**attempt)
                    logger.warning(
                        f"Discord alert failed (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Discord alert failed after {self.MAX_RETRIES} attempts, "
            f"discarding alert: {last_error}"
        )

    async def _send_dm(self, frame, num_people: int, timestamp: str, duration: float):
        """Send a Discord DM using the persistent connection.

        Args:
            frame: OpenCV frame (already copied)
            num_people: Number of people detected
            timestamp: Timestamp string
            duration: How long the face was present before alert (seconds)

        Raises:
            RuntimeError: If DM channel not initialized
            discord.errors.*: Various Discord API errors
        """
        import discord

        if self._dm_channel is None:
            raise RuntimeError("Discord DM channel not initialized")

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = io.BytesIO(buffer.tobytes())
        image_bytes.seek(0)

        message = (
            f"**Security Alert**\n"
            f"Time: {timestamp}\n"
            f"People detected: {num_people}\n"
            f"Present for: {duration:.1f}s before alert"
        )

        await self._dm_channel.send(
            content=message,
            file=discord.File(image_bytes, filename="alert.jpg"),
        )

        logger.info("Discord alert sent")

    def should_alert(self, current_count: int) -> bool:
        """Check if alert should be sent.

        Alerts only when going from 0 -> 1+ people, respecting cooldown.

        Args:
            current_count: Current number of people detected

        Returns:
            bool: True if alert should be sent
        """
        # Only alert when going from 0 -> 1+
        if self.previous_count > 0 or current_count == 0:
            return False

        # Check cooldown
        now = time.time()
        if now - self.last_alert_time < self.cooldown_seconds:
            return False

        return True

    def send_alert(self, frame, num_people: int, duration: float = 0.0) -> None:
        """Queue alert for async sending (non-blocking).

        Makes a copy of the frame since the original buffer may be reused.

        Args:
            frame: OpenCV frame with detections drawn
            num_people: Number of people detected
            duration: How long the face was present before alert (seconds)
        """
        # Copy frame since original buffer gets reused
        frame_copy = frame.copy()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self._queue.put_nowait((frame_copy, num_people, timestamp, duration))
            # Update state immediately (alert is queued)
            self.last_alert_time = time.time()
            self.previous_count = num_people
        except queue.Full:
            logger.warning("Alert queue full, discarding alert")

    def update_count(self, current_count: int) -> None:
        """Update previous count (call each frame).

        Args:
            current_count: Current number of people detected
        """
        self.previous_count = current_count

    def shutdown(self) -> None:
        """Stop the worker thread and disconnect the Discord bot."""
        self._shutdown_event.set()

        # Wake up the queue processor
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        # Close the Discord client from the main thread
        if self._loop and self._client and not self._client.is_closed():
            future = asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass

        self._worker_thread.join(timeout=5)
        if self._worker_thread.is_alive():
            logger.warning("Discord alert worker thread did not shut down cleanly")
