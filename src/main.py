"""
main.py – Full CCTV Assistant pipeline orchestrator.

Pipeline:
    Camera stream → Frame processing → YOLO detection → Face recognition →
    Action inference → Event generation → Event logging → Vector embedding →
    Alert evaluation.

The Telegram bot runs in a parallel thread.
"""

import sys
import time
import signal
import logging
import threading
from pathlib import Path
import cv2

# ── ensure project root on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, load_env, setup_logger, resolve_path
from src.camera.camera_stream import CameraStream
from src.camera.frame_processor import FrameProcessor
from src.detection.object_detector import ObjectDetector
from src.detection.face_recognition_module import FaceRecognizer
from src.detection.action_detector import ActionDetector
from src.events.event_generator import EventGenerator
from src.events.event_logger import EventLogger
from src.memory.event_embedding import EventEmbedder
from src.memory.semantic_search import SemanticSearch
from src.chatbot.query_engine import QueryEngine
from src.chatbot.response_generator import ResponseGenerator
from src.chatbot.telegram_bot import TelegramBot
from src.alerts.alert_manager import AlertManager

from database.db_manager import DatabaseManager
from vector_store.faiss_index import FaissIndex
from vector_store.embedding_manager import EmbeddingManager


# ── Graceful shutdown ──────────────────────────────────────────────────
_shutdown = threading.Event()


def _signal_handler(sig, frame):
    print("\n⏹  Shutting down…")
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_telegram_bot(bot: TelegramBot) -> None:
    """Start the Telegram bot in a separate thread."""
    try:
        bot.run()
    except Exception as exc:
        logging.getLogger("cctv_assistant").error(
            "Telegram bot error: %s", exc, exc_info=True
        )


def main() -> None:
    # ── 1. Configuration ──────────────────────────────────────────────
    load_env()
    config = load_config()
    logger = setup_logger(config=config)
    logger.info("=" * 60)
    logger.info("  Smart AI CCTV Assistant – Starting")
    logger.info("=" * 60)

    # ── 2. Database ───────────────────────────────────────────────────
    db_path = resolve_path(config["database"]["path"])
    db = DatabaseManager(db_path)
    db.initialize()
    logger.info("Database initialized: %s", db_path)

    # ── 3. Vector store & embeddings ──────────────────────────────────
    vs_cfg = config.get("vector_store", {})
    emb_manager = EmbeddingManager(model_name=vs_cfg.get("embedding_model", "all-MiniLM-L6-v2"))

    faiss_idx = FaissIndex(
        dimension=384,
        index_path=str(resolve_path(vs_cfg.get("index_path", "vector_store/faiss_store.index"))),
        metadata_path=str(resolve_path(vs_cfg.get("metadata_path", "vector_store/event_metadata.pkl"))),
    )
    faiss_idx.load()

    event_embedder = EventEmbedder(emb_manager, faiss_idx, auto_save_interval=5)

    # ── 4. Detection components ───────────────────────────────────────
    det_cfg = config.get("detection", {})
    object_detector = ObjectDetector(
        model_path=det_cfg.get("yolo_model", "yolov8n.pt"),
        conf_threshold=det_cfg.get("confidence_threshold", 0.5),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        target_classes=det_cfg.get("target_classes"),
    )

    face_cfg = config.get("face_recognition", {})
    face_recognizer = FaceRecognizer(
        known_faces_dir=str(resolve_path(face_cfg.get("known_faces_dir", "data/faces"))),
        encodings_file=str(resolve_path(face_cfg.get("encodings_file", "models/face/known_faces.pkl"))),
        tolerance=face_cfg.get("tolerance", 0.5),
        model=face_cfg.get("model", "hog"),
    )
    face_recognizer.load_known_faces()

    cam_cfg = config.get("camera", {})
    action_detector = ActionDetector(
        frame_width=cam_cfg.get("frame_width", 640),
        frame_height=cam_cfg.get("frame_height", 480),
    )

    # ── 5. Event pipeline ─────────────────────────────────────────────
    evt_cfg = config.get("events", {})
    event_generator = EventGenerator(
        dedup_window=evt_cfg.get("deduplication_window_seconds", 30),
    )
    event_logger = EventLogger(db_manager=db, event_embedder=event_embedder)

    # ── 6. Alert system ───────────────────────────────────────────────
    semantic_search = SemanticSearch(emb_manager, faiss_idx)
    response_gen = ResponseGenerator(
        provider=config.get("llm", {}).get("provider", "openai"),
        model=config.get("llm", {}).get("model", "gpt-3.5-turbo"),
        temperature=config.get("llm", {}).get("temperature", 0.3),
        max_tokens=config.get("llm", {}).get("max_tokens", 300),
    )
    query_engine = QueryEngine(semantic_search, response_gen, db_manager=db)

    telegram_bot = TelegramBot(query_engine=query_engine, db_manager=db)
    alert_manager = AlertManager(telegram_bot=telegram_bot, config=config.get("alerts", {}))

    # ── 7. Start Telegram bot in background ───────────────────────────
    bot_thread = threading.Thread(target=run_telegram_bot, args=(telegram_bot,), daemon=True)
    bot_thread.start()
    logger.info("Telegram bot started in background thread")

    # ── 8. Camera & frame processor ───────────────────────────────────
    camera = CameraStream(
        source=str(cam_cfg.get("stream_url", "0")),
        width=cam_cfg.get("frame_width", 640),
        height=cam_cfg.get("frame_height", 480),
        fps=cam_cfg.get("fps", 15),
    )
    frame_proc = FrameProcessor(
        skip_frames=cam_cfg.get("skip_frames", 5),
        clip_duration=evt_cfg.get("clip_duration_seconds", 8),
        clip_save_dir=str(resolve_path(evt_cfg.get("clip_save_dir", "data/recordings"))),
        fps=cam_cfg.get("fps", 15),
    )

    # ── 9. Main processing loop ───────────────────────────────────────
    logger.info("Starting camera stream…")
    frame_count = 0
    processed_count = 0
    loop_start = time.time()

    with camera:
        logger.info("Camera stream started")

        for frame in camera.stream_frames():

            if _shutdown.is_set():
                break

            if frame is None:
                logger.warning("Empty frame received, skipping...")
                continue

            frame_count += 1

            # Heartbeat log every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - loop_start
                logger.info(
                    "📹  Heartbeat: %d frames received, %d processed  (%.1fs elapsed)",
                    frame_count,
                    processed_count,
                    elapsed,
                )

            # Buffer frame for clip saving
            frame_proc.add_to_buffer(frame)

            if not frame_proc.should_process():
                cv2.imshow("CCTV AI", frame)
                cv2.waitKey(1)
                continue

            processed_count += 1
            processed = frame_proc.preprocess(frame)
            cv2.imshow("CCTV AI", processed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
              break
            # Object detection
            detections = object_detector.detect(processed)
            logger.debug(
                "[Frame %d] YOLO detections: %d objects",
                frame_count,
                len(detections) if detections else 0,
            )

            persons = object_detector.detect_persons(detections)
            objects = object_detector.detect_objects(detections)

            if not persons:
                logger.debug("[Frame %d] No person detected", frame_count)
                continue

            logger.info(
                "[Frame %d] %d person(s) detected",
                frame_count,
                len(persons),
            )

            # Face recognition
            face_results = face_recognizer.recognize_faces(processed)
            logger.debug(
                "[Frame %d] Face recognition: %d face(s)",
                frame_count,
                len(face_results) if face_results else 0,
            )

            # Action inference
            actions = action_detector.detect_actions(persons, objects, face_results)

            if not actions:
                logger.debug("[Frame %d] No action detected", frame_count)
                actions = [{"type": "person_detected"}]

            logger.info(
                "[Frame %d] %d action(s) detected: %s",
                frame_count,
                len(actions),
                actions,
            )

            # Save video clip
            clip_path = frame_proc.save_clip(tag="event")

            events = event_generator.generate_events(actions, clip_path=clip_path)

            if events:
                logger.info(
                    "[Frame %d] %d event(s) generated: %s",
                    frame_count,
                    len(events),
                    [e.get("summary", e) for e in events],
                )

                event_logger.log_events(events)
                alert_manager.send_alerts_sync(events)

    # ── Cleanup ───────────────────────────────────────────────────────
    event_embedder.flush()
    db.close()
    logger.info(
        "CCTV Assistant stopped.  Total frames: %d, processed: %d",
        frame_count,
        processed_count,
    )


if __name__ == "__main__":
    main()
