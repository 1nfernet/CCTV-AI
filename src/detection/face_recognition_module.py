"""
face_recognition_module.py – Load known face encodings and match against detected faces.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("cctv_assistant")


class FaceRecognizer:
    """
    Uses the ``face_recognition`` library to:

    1. Load pre-computed encodings of known faces from a pickle file.
    2. Detect faces in a frame and compare them to the known encodings.
    3. Return identity labels with confidence scores.

    Parameters
    ----------
    known_faces_dir : str | Path
        Directory containing sub-folders named after each person,
        each holding reference images (used by ``train_face_embeddings.py``).
    encodings_file : str | Path
        Path to the pickled encodings file.
    tolerance : float
        Distance threshold for a match (lower = stricter).
    model : str
        ``"hog"`` (fast, CPU) or ``"cnn"`` (accurate, GPU).
    """

    def __init__(
        self,
        known_faces_dir: str | Path = "data/faces",
        encodings_file: str | Path = "models/face/known_faces.pkl",
        tolerance: float = 0.5,
        model: str = "hog",
    ):
        self.known_faces_dir = Path(known_faces_dir)
        self.encodings_file = Path(encodings_file)
        self.tolerance = tolerance
        self.model = model

        # Populated after load_known_faces()
        self.known_encodings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self._fr: Any = None                    # lazy-imported module

    # ── lazy import ────────────────────────────────────────────────────
    def _import_fr(self):
        if self._fr is None:
            import face_recognition as fr
            self._fr = fr
        return self._fr

    # ── loading ────────────────────────────────────────────────────────
    def load_known_faces(self) -> None:
        """
        Load pre-computed encodings from ``encodings_file``.
        If the file does not exist, the system will still run but every
        face will be classified as *Unknown*.
        """
        if not self.encodings_file.exists():
            logger.warning(
                "Known-faces file not found: %s  – running without face recognition",
                self.encodings_file,
            )
            return

        with open(self.encodings_file, "rb") as fh:
            data = pickle.load(fh)

        self.known_encodings = data.get("encodings", [])
        self.known_names = data.get("names", [])
        logger.info(
            "Loaded %d known face(s): %s",
            len(self.known_names),
            ", ".join(set(self.known_names)),
        )

    # ── recognition ────────────────────────────────────────────────────
    def recognize_faces(self, frame: np.ndarray) -> list[dict]:
        """
        Detect and identify faces in a BGR frame.

        Returns
        -------
        list[dict]
            Each dict::

                {
                    "name": str,
                    "confidence": float,        # 0-100 %
                    "location": (top, right, bottom, left),
                }
        """
        fr = self._import_fr()

        # face_recognition expects RGB
        rgb = frame[:, :, ::-1]

        face_locations = fr.face_locations(rgb, model=self.model)
        if not face_locations:
            return []

        face_encodings = fr.face_encodings(rgb, face_locations)

        results: list[dict] = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            confidence = 0.0

            if self.known_encodings:
                distances = fr.face_distance(self.known_encodings, encoding)
                best_idx = int(np.argmin(distances))
                best_dist = distances[best_idx]

                if best_dist <= self.tolerance:
                    name = self.known_names[best_idx]
                    confidence = round((1.0 - best_dist) * 100, 1)

            results.append(
                {
                    "name": name,
                    "confidence": confidence,
                    "location": location,       # (top, right, bottom, left)
                }
            )

        return results
