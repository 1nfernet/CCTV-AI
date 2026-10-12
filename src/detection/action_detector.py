"""
action_detector.py – Proximity-based action inference from detections.

Combines person detections and object detections to produce human-readable
action descriptions like "opened fridge" or "watching TV".
"""

import logging
import math

logger = logging.getLogger("cctv_assistant")

# ── Zone / action mapping ──────────────────────────────────────────────
# Each rule maps a detected object class to an action and a location.
# ``proximity`` is the max pixel distance between person center and object center.
OBJECT_ACTION_RULES: list[dict] = [
    {"object": "refrigerator", "action": "opened the fridge",      "location": "kitchen",     "proximity": 200},
    {"object": "oven",         "action": "used the oven",          "location": "kitchen",     "proximity": 200},
    {"object": "microwave",    "action": "used the microwave",     "location": "kitchen",     "proximity": 200},
    {"object": "sink",         "action": "used the sink",          "location": "kitchen",     "proximity": 200},
    {"object": "tv",           "action": "watching TV",            "location": "living room", "proximity": 350},
    {"object": "laptop",       "action": "using laptop",           "location": "room",        "proximity": 200},
    {"object": "cell phone",   "action": "using phone",            "location": "room",        "proximity": 150},
    {"object": "book",         "action": "reading a book",         "location": "room",        "proximity": 150},
    {"object": "bottle",       "action": "picked up a bottle",     "location": "room",        "proximity": 150},
    {"object": "cup",          "action": "drinking from a cup",    "location": "room",        "proximity": 150},
    {"object": "bowl",         "action": "eating from a bowl",     "location": "kitchen",     "proximity": 150},
    {"object": "chair",        "action": "sitting on a chair",     "location": "room",        "proximity": 180},
    {"object": "couch",        "action": "sitting on the couch",   "location": "living room", "proximity": 250},
    {"object": "bed",          "action": "lying on the bed",       "location": "bedroom",     "proximity": 300},
    {"object": "dining table", "action": "at the dining table",    "location": "dining room", "proximity": 250},
    {"object": "knife",        "action": "using a knife",          "location": "kitchen",     "proximity": 150},
    {"object": "toothbrush",   "action": "brushing teeth",         "location": "bathroom",    "proximity": 150},
    {"object": "sandwich",     "action": "eating a sandwich",      "location": "kitchen",     "proximity": 150},
    {"object": "pizza",        "action": "eating pizza",           "location": "kitchen",     "proximity": 150},
    {"object": "banana",       "action": "eating a banana",        "location": "kitchen",     "proximity": 150},
    {"object": "apple",        "action": "eating an apple",        "location": "kitchen",     "proximity": 150},
]


def _distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class ActionDetector:
    """
    Infers actions by measuring the proximity between each detected person
    and every detected object, then applying ``OBJECT_ACTION_RULES``.

    If no object rule matches, the person is simply reported as *present*
    (with a heuristic location based on their position in the frame).
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.frame_width = frame_width
        self.frame_height = frame_height

    # ── location heuristic ─────────────────────────────────────────────
    def _infer_location_from_position(self, center: tuple) -> str:
        """
        Rough zone guess based on where in the frame the person appears.
        When cameras are positioned at an entrance the left/right edges
        often correspond to the doorway.
        """
        cx, cy = center
        rel_x = cx / max(self.frame_width, 1)

        if rel_x < 0.15 or rel_x > 0.85:
            return "entrance"
        return "room"

    # ── main API ───────────────────────────────────────────────────────
    def detect_actions(
        self,
        persons: list[dict],
        objects: list[dict],
        face_results: list[dict] | None = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        persons : list[dict]
            Person detections from ``ObjectDetector`` (must have ``center``).
        objects : list[dict]
            Non-person detections from ``ObjectDetector``.
        face_results : list[dict] | None
            Results from ``FaceRecognizer.recognize_faces``.

        Returns
        -------
        list[dict]
            Each dict::

                {
                    "person": str,
                    "action": str,
                    "object": str,
                    "location": str,
                    "confidence": float,
                }
        """
        actions: list[dict] = []

        for idx, person in enumerate(persons):
            person_center = person["center"]

            # Try to find identity from face results
            identity = "Unknown"
            face_conf = 0.0
            if face_results:
                # Match face to person by proximity (face location is top,right,bottom,left)
                for fr in face_results:
                    top, right, bottom, left = fr["location"]
                    face_center = ((left + right) / 2, (top + bottom) / 2)
                    if _distance(person_center, face_center) < 200:
                        identity = fr["name"]
                        face_conf = fr["confidence"]
                        break

            # Check proximity to objects
            matched = False
            for obj in objects:
                obj_center = obj["center"]
                dist = _distance(person_center, obj_center)

                for rule in OBJECT_ACTION_RULES:
                    if obj["class"] == rule["object"] and dist <= rule["proximity"]:
                        actions.append(
                            {
                                "person": identity,
                                "action": rule["action"],
                                "object": obj["class"],
                                "location": rule["location"],
                                "confidence": obj["confidence"],
                            }
                        )
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                # Default: person detected in an inferred zone
                loc = self._infer_location_from_position(person_center)
                action = "entered the house" if loc == "entrance" else "is present"
                actions.append(
                    {
                        "person": identity,
                        "action": action,
                        "object": "",
                        "location": loc,
                        "confidence": person["confidence"],
                    }
                )

        return actions
