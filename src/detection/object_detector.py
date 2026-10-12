"""
object_detector.py – YOLOv8-based object detection with class filtering.
"""

import logging
from typing import Any
import numpy as np

logger = logging.getLogger("cctv_assistant")


class ObjectDetector:
    """
    Wraps an Ultralytics YOLOv8 model for inference on individual frames.

    Only detections whose class name appears in ``target_classes`` (and
    whose confidence exceeds ``conf_threshold``) are returned.

    Parameters
    ----------
    model_path : str
        Path or Ultralytics model name (e.g. ``"yolov8n.pt"``).
    conf_threshold : float
        Minimum confidence score.
    iou_threshold : float
        IoU threshold for NMS.
    target_classes : list[str] | None
        Allowlist of COCO class names.  ``None`` = keep all classes.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: list[str] | None = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = set(target_classes) if target_classes else None
        self.model: Any = None

    def load_model(self) -> None:
        """Lazy-load the YOLO model."""
        from ultralytics import YOLO

        logger.info("Loading YOLO model: %s", self.model_path)
        self.model = YOLO(self.model_path)
        logger.info("YOLO model loaded successfully")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single frame.

        Returns
        -------
        list[dict]
            Each dict contains::

                {
                    "class": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],   # pixel coords
                    "center": (cx, cy),
                }
        """
        if self.model is None:
            self.load_model()

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                if self.target_classes and cls_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append(
                    {
                        "class": cls_name,
                        "confidence": round(conf, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": (int(cx), int(cy)),
                    }
                )

        return detections

    def detect_persons(self, detections: list[dict]) -> list[dict]:
        """Filter detections to only *person* class."""
        return [d for d in detections if d["class"] == "person"]

    def detect_objects(self, detections: list[dict]) -> list[dict]:
        """Filter detections to non-person classes."""
        return [d for d in detections if d["class"] != "person"]
