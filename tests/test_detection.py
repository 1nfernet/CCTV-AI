"""
test_detection.py – Unit tests for ObjectDetector and FaceRecognizer.

All tests use mocks so no GPU, camera, or model files are required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestObjectDetector(unittest.TestCase):
    """Tests for src.detection.object_detector.ObjectDetector."""

    def setUp(self):
        from src.detection.object_detector import ObjectDetector

        self.detector = ObjectDetector(
            model_path="yolov8n.pt",
            conf_threshold=0.5,
            target_classes=["person", "tv", "refrigerator"],
        )

    def test_detect_persons_filter(self):
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 100, 200], "center": (50, 100)},
            {"class": "tv", "confidence": 0.8, "bbox": [200, 100, 400, 300], "center": (300, 200)},
            {"class": "person", "confidence": 0.7, "bbox": [500, 0, 600, 200], "center": (550, 100)},
        ]
        persons = self.detector.detect_persons(detections)
        self.assertEqual(len(persons), 2)
        for p in persons:
            self.assertEqual(p["class"], "person")

    def test_detect_objects_filter(self):
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [0, 0, 100, 200], "center": (50, 100)},
            {"class": "tv", "confidence": 0.8, "bbox": [200, 100, 400, 300], "center": (300, 200)},
        ]
        objects = self.detector.detect_objects(detections)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["class"], "tv")


class TestActionDetector(unittest.TestCase):
    """Tests for src.detection.action_detector.ActionDetector."""

    def test_action_near_tv(self):
        from src.detection.action_detector import ActionDetector

        ad = ActionDetector(frame_width=640, frame_height=480)
        persons = [{"class": "person", "confidence": 0.9, "bbox": [250, 100, 350, 300], "center": (300, 200)}]
        objects = [{"class": "tv", "confidence": 0.8, "bbox": [280, 80, 450, 250], "center": (365, 165)}]

        actions = ad.detect_actions(persons, objects)
        self.assertTrue(len(actions) >= 1)
        self.assertEqual(actions[0]["action"], "watching TV")
        self.assertEqual(actions[0]["location"], "living room")

    def test_person_at_entrance(self):
        from src.detection.action_detector import ActionDetector

        ad = ActionDetector(frame_width=640, frame_height=480)
        persons = [{"class": "person", "confidence": 0.8, "bbox": [0, 50, 80, 400], "center": (40, 225)}]
        objects = []

        actions = ad.detect_actions(persons, objects)
        self.assertTrue(len(actions) >= 1)
        self.assertEqual(actions[0]["action"], "entered the house")
        self.assertEqual(actions[0]["location"], "entrance")


class TestFaceRecognizerInit(unittest.TestCase):
    """Basic initialisation tests for FaceRecognizer (no model loaded)."""

    def test_no_encodings_file(self):
        from src.detection.face_recognition_module import FaceRecognizer

        fr = FaceRecognizer(encodings_file="/nonexistent/path.pkl")
        fr.load_known_faces()   # should not raise
        self.assertEqual(fr.known_encodings, [])

    def test_empty_recognize(self):
        from src.detection.face_recognition_module import FaceRecognizer

        fr = FaceRecognizer(encodings_file="/nonexistent/path.pkl")
        fr.load_known_faces()
        # Mock the face_recognition import to avoid needing dlib
        mock_fr = MagicMock()
        mock_fr.face_locations.return_value = []
        fr._fr = mock_fr

        results = fr.recognize_faces(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
