"""
frame_processor.py – Frame pre-processing in the pipeline and video clip buffering.
"""

import cv2
import logging
import time
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

logger = logging.getLogger("cctv_assistant")


class FrameProcessor:
    """
    Handles frame-level operations:

    * **Skip logic** – only process every *N*-th frame.
    * **Resize** – resize frames for faster inference.
    * **Clip buffer** – keep a rolling buffer of recent frames so we can
      write a short video clip when an event fires.

    Parameters
    ----------
    skip_frames : int
        Process every *N*-th frame (1 = process every frame).
    clip_duration : int
        Length in seconds of saved video clips.
    clip_save_dir : str | Path
        Directory for saved clips.
    fps : int
        Frames-per-second used to compute buffer size.
    """

    def __init__(
        self,
        skip_frames: int = 5,
        clip_duration: int = 8,
        clip_save_dir: str | Path = "data/recordings",
        fps: int = 15,
    ):
        self.skip_frames = max(1, skip_frames)
        self.clip_duration = clip_duration
        self.clip_save_dir = Path(clip_save_dir)
        self.clip_save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps

        buffer_size = self.clip_duration * self.fps
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._frame_count = 0

    # ── frame gating ──────────────────────────────────────────────────
    def should_process(self) -> bool:
        """Return ``True`` if the current frame should be processed."""
        self._frame_count += 1
        return self._frame_count % self.skip_frames == 0

    def preprocess(self, frame: np.ndarray, target_width: int = 640) -> np.ndarray:
        """
        Resize ``frame`` proportionally to ``target_width`` pixels wide.
        """
        h, w = frame.shape[:2]
        if w == target_width:
            return frame
        scale = target_width / w
        new_h = int(h * scale)
        return cv2.resize(frame, (target_width, new_h))

    # ── clip buffering ────────────────────────────────────────────────
    def add_to_buffer(self, frame: np.ndarray) -> None:
        """Append a frame to the rolling clip buffer."""
        self._buffer.append(frame.copy())

    def save_clip(self, tag: str = "event") -> str:
        """
        Write the current buffer to a ``.mp4`` file and return the file path.

        Parameters
        ----------
        tag : str
            Short label used in the filename.

        Returns
        -------
        str   Absolute path to the saved clip.
        """
        if not self._buffer:
            logger.warning("Clip buffer is empty – nothing to save")
            return ""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{ts}.mp4"
        filepath = self.clip_save_dir / filename

        h, w = self._buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (w, h))

        for f in self._buffer:
            writer.write(f)
        writer.release()

        logger.info("Saved video clip: %s  (%d frames)", filepath, len(self._buffer))
        return str(filepath)

    def clear_buffer(self) -> None:
        """Explicitly clear the frame buffer."""
        self._buffer.clear()
