# 🎯 Smart AI CCTV Assistant

An intelligent home surveillance system that analyzes live camera feeds, detects people and objects, recognizes known faces, converts detections into structured events, stores them in a database, and lets you ask questions through a **Telegram chatbot**.

---

## 🧠 How It Works

```
Camera Stream → Frame Processing → YOLOv8 Object Detection → Face Recognition
→ Action Inference → Event Generation → Event Logging (SQLite + FAISS)
→ Alert Evaluation → Telegram Notifications
```

**Example Interactions:**

> **You:** "Did my sister come home?"  
> **Bot:** "Yes. She entered the house at 7:12 PM and went to the kitchen at 7:14 PM."

> **You:** "What happened today?"  
> **Bot:** "Sister entered the house at 7:12 PM, opened the fridge at 7:14 PM, and watched TV at 7:18 PM."

---

## 📁 Project Structure

```
CCTV-AI/
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (tokens, keys)
├── .gitignore                        # Git ignore rules
│
├── configs/
│   └── config.yaml                   # All system configuration
│
├── data/
│   ├── faces/                        # Known face images (sub-folders per person)
│   ├── recordings/                   # Saved video clips on events
│   └── logs/                         # Application logs
│
├── database/
│   ├── events.db                     # SQLite database (auto-created)
│   └── db_manager.py                 # Database CRUD wrapper
│
├── models/
│   ├── yolo/                         # YOLOv8 model weights (auto-downloaded)
│   └── face/                         # Face encoding pickle files
│
├── vector_store/
│   ├── faiss_index.py                # FAISS vector index wrapper
│   └── embedding_manager.py          # Sentence-transformer encoder
│
├── src/
│   ├── main.py                       # Main pipeline orchestrator
│   │
│   ├── camera/
│   │   ├── camera_stream.py          # RTSP / webcam capture
│   │   └── frame_processor.py        # Frame resize, skip, clip buffering
│   │
│   ├── detection/
│   │   ├── object_detector.py        # YOLOv8 inference
│   │   ├── face_recognition_module.py # Face detection & matching
│   │   └── action_detector.py        # Proximity-based action inference
│   │
│   ├── events/
│   │   ├── event_generator.py        # Detection → structured events
│   │   └── event_logger.py           # Persist to DB + vector store
│   │
│   ├── memory/
│   │   ├── event_embedding.py        # Event text → FAISS embedding
│   │   └── semantic_search.py        # Query → embedding → search
│   │
│   ├── chatbot/
│   │   ├── telegram_bot.py           # Bot commands & alert sending
│   │   ├── query_engine.py           # Semantic + DB search orchestration
│   │   └── response_generator.py     # LLM answer generation
│   │
│   ├── alerts/
│   │   └── alert_manager.py          # Rule engine for proactive alerts
│   │
│   └── utils/
│       ├── helpers.py                # Config loader, logger, path utils
│       └── time_utils.py             # Timestamp formatting
│
├── scripts/
│   ├── initialize_database.py        # Create the SQLite events table
│   └── train_face_embeddings.py      # Generate face encodings from photos
│
└── tests/
    ├── test_detection.py             # Object detection & face rec tests
    └── test_database.py              # Database CRUD tests
```

---

## ⚙️ Installation

### 1. Clone & navigate

```bash
git clone https://github.com/amangupta982/CCTV-AI.git
cd CCTV-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Edit `.env` with your credentials:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

### 5. Configure settings

Edit `configs/config.yaml`:

- Set `camera.stream_url` to your RTSP URL or `"0"` for webcam.
- Adjust detection thresholds, alert settings, etc.

---

## 🚀 Running

### Step 1: Initialize the database

```bash
python scripts/initialize_database.py
```

### Step 2: (Optional) Train face embeddings

Place photos in `data/faces/<PersonName>/` and run:

```bash
python scripts/train_face_embeddings.py
```

### Step 3: Start the system

```bash
python src/main.py
```

This starts:
- 📹 Camera capture & processing loop
- 🤖 Telegram bot (background thread)
- 🔔 Alert system

---

## 🤖 Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/ask <question>` | Ask a natural-language question about events |
| `/today` | Show all events from today |
| `/status` | Show system status |
| `/help` | List available commands |

---

## 🔔 Alert System

The system automatically sends Telegram alerts for:

| Trigger | Example |
|---------|---------|
| Unknown person detected | ⚠️ Alert: Unknown person detected at 2:15 AM |
| Night-time movement | 🌙 Alert: Movement detected at night (2:15 AM) |
| Door / entrance activity | 🚪 Alert: Sister detected at the entrance at 7:12 PM |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Tests use mocks and in-memory databases — no GPU or camera required.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Camera      │────▶│  YOLOv8         │────▶│  Face            │
│  Stream      │     │  Detection      │     │  Recognition     │
└──────────────┘     └─────────────────┘     └──────────────────┘
                              │                        │
                              ▼                        ▼
                     ┌─────────────────┐     ┌──────────────────┐
                     │  Action         │◀────│  Identity        │
                     │  Detector       │     │  Matching        │
                     └────────┬────────┘     └──────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       Event Generator         │
              │   (deduplication + summary)    │
              └───────────────┬───────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
             ┌──────────┐ ┌───────┐ ┌──────────┐
             │  SQLite  │ │ FAISS │ │  Alert   │
             │  Events  │ │ Index │ │  Manager │
             └──────────┘ └───────┘ └──────────┘
                    │         │         │
                    ▼         ▼         ▼
              ┌───────────────────────────────┐
              │       Telegram Chatbot        │
              │   /ask  /today  /status       │
              └───────────────────────────────┘
```

---

## 📝 Technology Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Face Recognition | face_recognition + dlib |
| Database | SQLite |
| Vector Search | FAISS + sentence-transformers |
| LLM | OpenAI GPT / local fallback |
| Chat Interface | python-telegram-bot |
| Camera Input | OpenCV |

---

## 📄 License

This project is for educational and personal use.

---

*Built with ❤️ as a Smart AI CCTV Assistant.*
