"""
camera_stream.py – Capture frames from RTSP streams or the laptop webcam.
"""

import cv2
import time
import logging
from typing import Generator
import numpy as np

logger = logging.getLogger("cctv_assistant")

# Maximum consecutive frame-read failures before giving up.
DEFAULT_MAX_RETRIES = 30          # ~3 seconds at 0.1 s sleep
RETRY_SLEEP_SECONDS = 0.1


class CameraStream:
    """
    Opens an OpenCV ``VideoCapture`` on an RTSP URL or local webcam index
    and yields frames one at a time.

    Parameters
    ----------
    source : str
        An RTSP URL (``rtsp://…``) or a webcam index as a string (``"0"``).
    width  : int
        Desired capture width.
    height : int
        Desired capture height.
    fps    : int
        Desired capture FPS (best-effort).
    max_retries : int
        How many consecutive frame failures to tolerate before stopping.
    """

    def __init__(
        self,
        source: str = "0",
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.source = int(source) if source.isdigit() else source
        self.width = width
        self.height = height
        self.fps = fps
        self.max_retries = max_retries
        self.cap: cv2.VideoCapture | None = None

    # ── lifecycle ──────────────────────────────────────────────────────
    def open(self) -> bool:
        """Open the video source.  Returns ``True`` on success."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error("Failed to open camera source: %s", self.source)
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info(
            "Camera opened  source=%s  %dx%d@%dfps",
            self.source,
            self.width,
            self.height,
            self.fps,
        )
        return True

    def release(self) -> None:
        """Release the capture device."""
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # ── frame iteration ────────────────────────────────────────────────
    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame.  Returns ``(success, frame)``."""
        if not self.is_opened():
            return False, None
        ret, frame = self.cap.read()  # type: ignore[union-attr]
        return ret, frame

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Infinite generator that yields frames until ``max_retries``
        consecutive failures or the capture is released.

        On a transient failure the generator sleeps briefly and retries
        instead of terminating the stream immediately.
        """
        if not self.is_opened():
            if not self.open():
                return

        consecutive_failures = 0
        total_frames = 0

        while True:
            ret, frame = self.read_frame()

            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(
                    "Frame read failed – retrying (%d/%d)…",
                    consecutive_failures,
                    self.max_retries,
                )
                if consecutive_failures >= self.max_retries:
                    logger.error(
                        "Reached %d consecutive frame failures – ending stream",
                        self.max_retries,
                    )
                    break
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            # Successful read — reset failure counter
            if consecutive_failures > 0:
                logger.info(
                    "Frame read recovered after %d failed attempt(s)",
                    consecutive_failures,
                )
            consecutive_failures = 0
            total_frames += 1

            if total_frames % 100 == 0:
                logger.debug("Camera stream: %d frames read so far", total_frames)

            yield frame

    # ── context manager ────────────────────────────────────────────────
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
