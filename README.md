# AstroLens

**AI-powered image analysis with ML classification, anomaly detection, and LLM-assisted insights.**

![Status](https://img.shields.io/badge/status-alpha-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)

---

## Overview

AstroLens is a local-first desktop application for intelligent image analysis:

- **Upload images** (FITS, PNG, JPEG) via drag-and-drop or file picker
- **Classify** objects using a fine-tuned Vision Transformer (ViT)
- **Detect anomalies** with energy-based out-of-distribution detection
- **Find similar** images using embedding similarity (FAISS)
- **Get LLM insights** via natural language (OpenAI or local Ollama)
- **Chat with an AI agent** to explore your image collection

All processing runs **locally** – no cloud required. Optional LLM features work with OpenAI API or fully offline with Ollama.

---

## Features

| Feature | Description |
|---------|-------------|
| **Image Upload** | Support for FITS, PNG, JPEG. Drag-and-drop or browse. |
| **ML Classification** | ViT-B/16 classifies into 10 galaxy morphology categories |
| **Anomaly Detection** | **Ensemble OOD** with Energy, MSP, and Mahalanobis distance voting |
| **Autonomous Discovery** | Background loop downloads, analyzes, and hunts for anomalies 24/7 |
| **Fine-Tuning** | Continuous model improvement with rotating datasets |
| **Embedding Search** | 768-dim vectors + FAISS for "find similar" |
| **Duplicate Detection** | Perceptual hashing prevents duplicate images |
| **Active Learning** | Flags uncertain samples for human review |
| **LLM Annotation** | GPT-4o or Ollama generates descriptions and hypotheses |
| **Chat Agent** | LangChain agent with tools: analyze, search, annotate, export |
| **Desktop UI** | Elegant PyQt5 interface with dark theme |
| **GPU Acceleration** | Apple MPS support for fast inference on Mac |
| **Containerized** | Docker Compose for one-command startup |
| **API-first** | FastAPI backend; UI is just a client |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone
git clone https://github.com/samantaba/AstroLens.git
cd AstroLens

# Start API backend
docker-compose up -d

# Run PyQt5 UI (requires Python on host)
pip install -r requirements.txt
python -m ui.main
```

### Option 2: Local Python

```bash
# Clone and setup
git clone https://github.com/samantaba/AstroLens.git
cd AstroLens
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download ML weights (first run)
python scripts/download_weights.py

# Start API
uvicorn api.main:app --reload --port 8000

# In another terminal, start UI
python -m ui.main
```

### Option 3: Fully Offline (with Ollama)

```bash
# Install Ollama
brew install ollama  # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh  # Linux

# Pull vision model
ollama pull llava

# Start AstroLens with local LLM
export LLM_PROVIDER=ollama
docker-compose up -d
python -m ui.main
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PyQt5 DESKTOP UI                               │
│  Gallery │ Viewer │ Analysis │ Chat │ Settings                      │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │ HTTP
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                                 │
│  /images │ /analysis │ /annotate │ /chat │ /candidates             │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   ML INFERENCE   │   │   LLM ANNOTATOR  │   │  LANGCHAIN AGENT │
│   (ViT + OOD)    │   │   (GPT/Ollama)   │   │   (tools)        │
└──────────────────┘   └──────────────────┘   └──────────────────┘
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  SQLite (metadata) │ Local Files (images/) │ FAISS (embeddings)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Autonomous Discovery Loop

AstroLens includes an autonomous discovery system that continuously:

1. **Downloads** images from multiple sources (SDSS, ZTF, NASA APOD, ESO)
2. **Analyzes** each image with ensemble OOD detection
3. **Flags** potential anomalies for review
4. **Fine-tunes** the model periodically to improve detection

### Running the Discovery Loop

```bash
# Start from UI: Settings → Discovery Loop → Start

# Or from command line:
python scripts/discovery_loop.py --turbo  # Fast mode (no delays)
python scripts/discovery_loop.py --aggressive  # Lower thresholds
```

### Ensemble Anomaly Detection

The detection system uses **3 complementary methods** that vote:

| Method | What It Measures |
|--------|------------------|
| **Energy Score** | Overall model confidence (higher = more unusual) |
| **MSP (Max Softmax Prob)** | Highest class probability (lower = more uncertain) |
| **Mahalanobis Distance** | Distance from known class distributions |

An image is flagged as anomaly if **2+ methods agree** it's out-of-distribution.

---

## ML vs LLM: What Each Does

| | **ML (Machine Learning)** | **LLM (Large Language Model)** |
|--|---------------------------|--------------------------------|
| **Purpose** | Classify images, detect anomalies | Generate text, answer questions |
| **Model** | ViT-B/16 (fine-tuned) | GPT-4o / LLaVA |
| **Runs** | Always locally | OpenAI API or local Ollama |
| **We train?** | No – use pre-trained weights | No – use via API |
| **Output** | Class labels, scores, vectors | Natural language text |

---

## Configuration

Create a `.env` file (optional):

```bash
# LLM Provider: "openai", "ollama", or "none"
LLM_PROVIDER=openai

# OpenAI API key (if using openai)
OPENAI_API_KEY=sk-...

# Ollama URL (if using ollama)
OLLAMA_URL=http://localhost:11434

# ML Settings
OOD_THRESHOLD=10.0
```

---

## Project Structure

```
AstroLens/
├── api/                    # FastAPI backend
│   ├── main.py             # App entry
│   ├── routes/             # Endpoints
│   ├── models.py           # Pydantic schemas
│   └── db.py               # SQLite via SQLAlchemy
├── inference/              # ML models
│   ├── classifier.py       # ViT classifier
│   ├── ood.py              # Anomaly detection
│   └── embeddings.py       # FAISS vector store
├── annotator/              # LLM layer
│   ├── prompts.py          # Prompt templates
│   └── chain.py            # LangChain annotator
├── agent/                  # LangChain agent
│   ├── tools.py            # Agent tools
│   └── agent.py            # Agent definition
├── ui/                     # PyQt5 desktop app
│   ├── main.py             # Entry point
│   ├── main_window.py      # Main window
│   ├── gallery.py          # Image grid
│   ├── viewer.py           # Detail view
│   └── chat_panel.py       # LLM chat
├── data/                   # Local storage (gitignored)
├── weights/                # ML weights (gitignored)
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/images` | GET | List all images |
| `/images` | POST | Upload image |
| `/images/{id}` | GET | Get image detail |
| `/analysis/classify/{id}` | POST | Classify image |
| `/analysis/anomaly/{id}` | POST | OOD detection |
| `/analysis/similar/{id}` | POST | Find similar |
| `/analysis/full/{id}` | POST | Run all analyses |
| `/annotate/{id}` | POST | LLM annotation |
| `/chat` | POST | Agent message |
| `/candidates` | GET | List anomalies |

Full API docs: `http://localhost:8000/docs`

---

## Maintainer

**Saman Tabatabaeian**  
Email: <saman.tabatabaeian@gmail.com>  
LinkedIn: [samantabatabaeian](https://www.linkedin.com/in/samantabatabaeian/)

---

## License

MIT License – see [LICENSE](LICENSE).
