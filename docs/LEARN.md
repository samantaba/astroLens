# 🔭 AstroLens Complete Learning Guide

A comprehensive guide to understanding, modifying, and extending the AstroLens codebase.

---

## Part 1: What is AstroLens?

AstroLens is an **AI-powered astronomical image analyzer**. It does three main things:

1. **Classifies images** - "Is this a spiral galaxy, elliptical galaxy, or something else?"
2. **Detects anomalies** - "Is this image unusual compared to what I've seen before?"
3. **Explains findings** - "Describe what you see in plain English"

The flow:
```
You upload an image → ML model says "spiral galaxy, 95% confident"
                    → Anomaly detector says "this looks normal"
                    → LLM says "This appears to be a barred spiral with prominent arms..."
```

---

## Part 2: The Three AI Layers

### Layer 1: Machine Learning (ML) - The "Eyes"

**What it does:** Pattern recognition. Looks at pixels and recognizes shapes.

**Model used:** Vision Transformer (ViT) - a neural network trained on millions of images.

**How it works:**
```
Image (256x256 pixels)
    ↓
Split into 16x16 patches (256 patches)
    ↓
Each patch becomes a vector (embedding)
    ↓
Transformer processes all patches together
    ↓
Output: "This looks like class 3 (spiral galaxy) with 87% confidence"
```

**Key file:** `inference/classifier.py`

**Why ViT?** Traditional CNNs (like ResNet) look at local patterns. ViT looks at the WHOLE image at once, which is better for astronomy where context matters.

---

### Layer 2: Out-of-Distribution (OOD) Detection - The "Surprise Detector"

**What it does:** Finds images that are DIFFERENT from the training data.

**How it works:**
```
Normal image → Model is confident → Low energy score → "Normal"
Weird image  → Model is confused  → High energy score → "ANOMALY!"
```

**The math:**
```python
energy = -log(sum(exp(logits)))  # Lower = more confident
```

If the model has seen many images like this before, it's confident (low energy).
If the image is unlike anything it's seen, it's uncertain (high energy).

**Key file:** `inference/ood.py`

**Why this matters:** You can't train a model on every possible anomaly. But you CAN detect when something doesn't fit the pattern.

---

### Layer 3: Large Language Model (LLM) - The "Explainer"

**What it does:** Writes human-readable descriptions.

**Models used:**
- **OpenAI GPT-4o** - Cloud API, best quality, costs money
- **Ollama LLaVA** - Runs locally, free, can see images
- **Ollama Llama3** - Runs locally, text-only

**How it works:**
```
System prompt: "You are an astronomer..."
+ Image description: "Classification: spiral galaxy, 87% confidence, anomaly score: -2.5"
+ User request: "Describe this image"
    ↓
LLM generates: "This appears to be a face-on spiral galaxy with two prominent arms..."
```

**Key files:** `annotator/chain.py`, `agent/agent.py`

---

## Part 3: Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Gallery    │  │    Viewer    │  │     Chat     │          │
│  │  (PyQt5)     │  │   (PyQt5)    │  │   (PyQt5)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP Requests
┌─────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                         │
│  /images      - Upload, list, delete images                     │
│  /analysis/*  - Run ML classification + anomaly detection       │
│  /annotate/*  - Generate LLM descriptions                       │
│  /chat        - Conversational agent                            │
│  /stats       - Collection statistics                           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   CLASSIFIER     │ │   OOD DETECTOR   │ │   LLM ANNOTATOR  │
│   (ViT Model)    │ │   (Energy-based) │ │   (OpenAI/Ollama)│
│                  │ │                  │ │                  │
│  inference/      │ │  inference/      │ │  annotator/      │
│  classifier.py   │ │  ood.py          │ │  chain.py        │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   SQLite     │  │    Images    │  │    FAISS     │          │
│  │  (metadata)  │  │  (files)     │  │  (vectors)   │          │
│  │  data/db.sql │  │  data/images │  │  data/faiss  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Folder Structure Explained

```
astroLens/
│
├── api/                    # BACKEND (FastAPI web server)
│   ├── main.py            # All HTTP endpoints defined here
│   ├── db.py              # Database operations (SQLAlchemy)
│   └── models.py          # Data structures (Pydantic schemas)
│
├── inference/              # ML MODELS
│   ├── classifier.py      # ViT image classification
│   ├── ood.py             # Anomaly detection
│   └── embeddings.py      # Vector similarity search
│
├── annotator/              # LLM INTEGRATION
│   ├── chain.py           # LangChain LLM calls
│   └── prompts.py         # Prompt templates
│
├── agent/                  # CONVERSATIONAL AI
│   ├── agent.py           # LangChain agent logic
│   └── tools.py           # Actions the agent can take
│
├── ui/                     # DESKTOP GUI (PyQt5)
│   ├── main.py            # App entry point
│   ├── main_window.py     # Main window layout
│   ├── gallery.py         # Image grid view
│   ├── viewer.py          # Single image view
│   └── chat_panel.py      # Chat interface
│
├── finetuning/            # MODEL TRAINING
│   ├── download_datasets.py  # Get training data
│   ├── train.py              # Fine-tune the model
│   └── evaluate.py           # Test model accuracy
│
├── scripts/               # UTILITIES
│   └── nightly_ingest.py  # Download images from surveys
│
├── data/                  # RUNTIME DATA (gitignored)
│   ├── images/            # Uploaded images
│   └── astrolens.db       # SQLite database
│
└── weights/               # MODEL WEIGHTS (gitignored)
    └── vit_astrolens/     # Fine-tuned model
```

---

## Part 5: Core Technologies Explained

### 5.1 FastAPI (Backend Framework)

**What is it?** A Python web framework for building APIs.

**Why FastAPI?**
- Fast (async support)
- Auto-generates documentation
- Type checking with Pydantic
- Easy to learn

**How endpoints work:**

```python
@app.post("/images")
async def upload_image(file: UploadFile = File(...)):
    # This function runs when someone POSTs to /images
    # 'file' is automatically parsed from the request
    
    # Save the file
    filepath = IMAGES_DIR / file.filename
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    # Add to database
    image = create_image(db, filename=file.filename, filepath=str(filepath))
    
    # Return JSON response
    return {"id": image.id, "filename": image.filename}
```

**Key concepts:**
- `@app.get("/path")` - Handle GET requests
- `@app.post("/path")` - Handle POST requests  
- `async def` - Function can pause while waiting for I/O
- `Depends(get_db)` - Inject database session

---

### 5.2 SQLAlchemy (Database ORM)

**What is it?** A library that lets you use Python objects instead of raw SQL.

**Why use it?**
- Write Python, not SQL
- Database-agnostic (works with SQLite, PostgreSQL, etc.)
- Automatic migrations

**How it works:**

```python
# Define a table as a Python class
class ImageRecord(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    class_label = Column(String)           # "spiral_galaxy"
    class_confidence = Column(Float)       # 0.87
    is_anomaly = Column(Boolean)           # True/False
    llm_description = Column(Text)         # "This appears to be..."

# Create a new record
image = ImageRecord(filename="galaxy.png")
db.add(image)
db.commit()

# Query records
anomalies = db.query(ImageRecord).filter(ImageRecord.is_anomaly == True).all()
```

**Key file:** `api/db.py`

---

### 5.3 PyQt5 (Desktop GUI)

**What is it?** A framework for building desktop applications with Python.

**Why PyQt5?**
- Native look and feel
- Cross-platform (Mac, Windows, Linux)
- Rich widget library

**How it works:**

```python
class GalleryPanel(QWidget):
    # Signals - emit events that other parts can listen to
    image_selected = pyqtSignal(int)  # Emits image ID when clicked
    
    def __init__(self):
        super().__init__()
        
        # Layout - arrange widgets
        layout = QVBoxLayout(self)
        
        # Widgets - UI elements
        title = QLabel("Image Gallery")
        layout.addWidget(title)
        
        button = QPushButton("Upload")
        button.clicked.connect(self.on_upload)  # Connect signal to slot
        layout.addWidget(button)
    
    def on_upload(self):
        # This runs when button is clicked
        files = QFileDialog.getOpenFileNames(self, "Select Images")
        # ... process files
```

**Key concepts:**
- `QWidget` - Base class for all UI elements
- `QLayout` - Arranges widgets (QVBoxLayout = vertical, QHBoxLayout = horizontal)
- `Signal/Slot` - Qt's event system (signal emits, slot receives)
- `QThread` - Run tasks in background without freezing UI

---

### 5.4 Hugging Face Transformers (ML Models)

**What is it?** A library for loading and using pre-trained AI models.

**Why Hugging Face?**
- Thousands of pre-trained models
- Easy to load and use
- Built-in fine-tuning support

**How it works:**

```python
from transformers import ViTForImageClassification, ViTImageProcessor

# Load model from Hugging Face Hub
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Or load your fine-tuned model
model = ViTForImageClassification.from_pretrained("weights/vit_astrolens")

# Process an image
image = Image.open("galaxy.png")
inputs = processor(images=image, return_tensors="pt")

# Get prediction
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(-1).item()
```

**Key file:** `inference/classifier.py`

---

### 5.5 LangChain (LLM Framework)

**What is it?** A framework for building applications with language models.

**Why LangChain?**
- Unified interface for different LLMs (OpenAI, Ollama, etc.)
- Memory management (conversation history)
- Tool/function calling

**How it works:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Create LLM instance
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Send messages
messages = [
    SystemMessage(content="You are an astronomer."),
    HumanMessage(content="What is a spiral galaxy?")
]

response = llm.invoke(messages)
print(response.content)  # "A spiral galaxy is..."
```

**Key files:** `annotator/chain.py`, `agent/agent.py`

---

## Part 6: Data Flow - Complete Walkthrough

### Flow 1: Uploading an Image

```
User drops image → UI catches drop event
                    ↓
                  main_window.py: dropEvent()
                    ↓
                  HTTP POST to /images
                    ↓
                  api/main.py: upload_image()
                    ↓
                  Save file to data/images/
                    ↓
                  Create database record
                    ↓
                  Return JSON {id: 1, filename: "galaxy.png"}
                    ↓
                  UI refreshes gallery
```

### Flow 2: Analyzing an Image

```
User clicks "Analyze" → HTTP POST to /analysis/full/{id}
                          ↓
                        api/main.py: full_analysis()
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    CLASSIFY           OOD CHECK        EMBED
    classifier.py      ood.py           embeddings.py
        ↓                 ↓                 ↓
    "spiral_galaxy"    score: -2.5      768-dim vector
    confidence: 87%    is_anomaly: F
        ↓                 ↓                 ↓
        └─────────────────┼─────────────────┘
                          ↓
                    Update database record
                          ↓
                    Return JSON results
                          ↓
                    UI updates display
```

### Flow 3: Chat Interaction

```
User types "analyze all images" → HTTP POST to /chat
                                    ↓
                                  api/main.py: chat_endpoint()
                                    ↓
                                  agent/agent.py: chat()
                                    ↓
                              Does message contain action keywords?
                                    ↓ YES
                              Run heuristic_response()
                                    ↓
                              Loop through unanalyzed images
                                    ↓
                              Call analyze_image tool for each
                                    ↓
                              Return summary "Analyzed 5 images..."
                                    ↓
                              UI displays in chat bubble
```

---

## Part 7: How to Modify Common Things

### 7.1 Add a New Galaxy Class

1. **Update training data:** Add images to `finetuning/datasets/my_class/`
2. **Re-train:** `python finetuning/train.py --dataset finetuning/datasets/`
3. **The model automatically learns the new class!**

### 7.2 Change the Anomaly Threshold

In `api/main.py`:
```python
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "10.0"))
```

Lower = more sensitive (more anomalies flagged)
Higher = less sensitive (only very unusual things flagged)

### 7.3 Add a New API Endpoint

In `api/main.py`:
```python
@app.get("/my-endpoint/{image_id}")
async def my_endpoint(image_id: int, db: Session = Depends(get_db)):
    image = get_image(db, image_id)
    # Do something
    return {"result": "success"}
```

### 7.4 Add a New UI Panel

1. Create `ui/my_panel.py`:
```python
class MyPanel(QWidget):
    def __init__(self, api):
        super().__init__()
        self.api = api
        # Build UI...
```

2. Add to `main_window.py`:
```python
from .my_panel import MyPanel

self.my_panel = MyPanel(self.api)
self.content_stack.addWidget(self.my_panel)  # Index 3
```

### 7.5 Add a New Chat Command

In `agent/agent.py`, add to `_heuristic_response()`:
```python
if "my command" in message_lower:
    # Do something
    return {"output": "Done!"}
```

### 7.6 Add a New Image Source

In `scripts/nightly_ingest.py`:
```python
def download_my_source(count: int, output_dir: Path) -> List[Path]:
    # Download images from your source
    # Return list of file paths
    pass
```

Then add to `main()`:
```python
if args.source in ["my_source", "all"]:
    downloads = download_my_source(args.count, output_base / "my_source")
    all_downloads.extend(downloads)
```

---

## Part 8: Key Code Segments Explained

### 8.1 The Classifier

```python
class AstroClassifier:
    def __init__(self, weights_path=None):
        # Load the model
        if weights_path and Path(weights_path).exists():
            # Load YOUR fine-tuned model
            self.model = ViTForImageClassification.from_pretrained(weights_path)
        else:
            # Load the default pre-trained model
            self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    
    def classify(self, image_path):
        # 1. Load image
        image = Image.open(image_path).convert("RGB")
        
        # 2. Preprocess (resize, normalize)
        inputs = self.processor(images=image, return_tensors="pt")
        
        # 3. Run through model
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # 4. Get class probabilities
        probs = torch.softmax(outputs.logits, dim=-1)
        
        # 5. Get top prediction
        top_idx = probs.argmax().item()
        confidence = probs[0, top_idx].item()
        
        return ClassificationOutput(
            class_label=self.class_names[top_idx],
            confidence=confidence,
            logits=outputs.logits,  # Raw scores for OOD
            embedding=outputs.hidden_states[-1][:, 0, :]  # For similarity
        )
```

### 8.2 The OOD Detector

```python
class OODDetector:
    def __init__(self, threshold=10.0):
        self.threshold = threshold
    
    def detect(self, logits):
        # Energy-based OOD detection
        # Lower energy = model is confident = normal
        # Higher energy = model is confused = anomaly
        
        energy = -torch.logsumexp(logits, dim=-1).item()
        
        # Compare to threshold
        is_anomaly = energy > self.threshold
        
        return AnomalyResult(
            ood_score=energy,
            is_anomaly=is_anomaly,
            threshold=self.threshold
        )
```

### 8.3 The API Endpoint

```python
@app.post("/analysis/full/{image_id}")
async def full_analysis(image_id: int, db: Session = Depends(get_db)):
    """Run complete analysis: classify + OOD + embed"""
    
    # 1. Get image from database
    image = get_image(db, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    
    # 2. Load ML models (lazy loading - only load once)
    classifier = get_classifier()
    ood_detector = get_ood_detector()
    
    # 3. Run classification
    result = classifier.classify(image.filepath)
    
    # 4. Run anomaly detection on the logits
    anomaly = ood_detector.detect(result.logits)
    
    # 5. Update database
    update_image(db, image_id,
        class_label=result.class_label,
        class_confidence=result.confidence,
        ood_score=anomaly.ood_score,
        is_anomaly=anomaly.is_anomaly
    )
    
    # 6. Return results
    return {
        "classification": {"class_label": result.class_label, "confidence": result.confidence},
        "anomaly": {"ood_score": anomaly.ood_score, "is_anomaly": anomaly.is_anomaly}
    }
```

### 8.4 The Chat Agent

```python
class AstroLensAgent:
    def chat(self, message, db=None):
        message_lower = message.lower()
        
        # ACTION requests - run directly, don't ask LLM
        if "analyze" in message_lower or "list" in message_lower:
            return self._heuristic_response(message, db)
        
        # QUESTIONS - ask the LLM
        if self.llm:
            response = self.llm.invoke([
                SystemMessage(content="You are AstroLens assistant..."),
                HumanMessage(content=message)
            ])
            return {"output": response.content}
        
        # Fallback
        return {"output": "I can help with: list images, analyze, statistics"}
    
    def _heuristic_response(self, message, db):
        """Handle action commands directly"""
        
        if "analyze" in message.lower():
            # Get unanalyzed images and analyze them
            results = []
            for img_id in unanalyzed_ids:
                result = analyze_image.invoke({"image_id": img_id})
                results.append(result)
            
            return {"output": f"Analyzed {len(results)} images"}
```

---

## Part 9: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_PATH` | `weights/vit_astrolens` | Path to fine-tuned model |
| `OOD_THRESHOLD` | `10.0` | Anomaly detection sensitivity |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | - | Required for OpenAI |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `IMAGES_DIR` | `data/images` | Where images are stored |

---

## Part 10: Common Operations

### Start the system:
```bash
# Terminal 1 - API
cd /path/to/astroLens
source .venv/bin/activate
export LLM_PROVIDER="ollama"
uvicorn api.main:app --reload --port 8000

# Terminal 2 - UI
python -m ui.main
```

### Download and analyze images:
```bash
python scripts/nightly_ingest.py --source sdss --count 50 --upload --analyze
```

### Re-train the model:
```bash
python finetuning/train.py --dataset finetuning/datasets/galaxy10 --epochs 5
```

### Check API health:
```bash
curl http://localhost:8000/health
```

---

## Part 11: Glossary

| Term | Meaning |
|------|---------|
| **ViT** | Vision Transformer - a neural network architecture for images |
| **OOD** | Out-of-Distribution - data that differs from training data |
| **Embedding** | A vector representation of an image (768 numbers) |
| **Logits** | Raw model outputs before softmax (used for OOD) |
| **Fine-tuning** | Training a pre-trained model on your specific data |
| **Epoch** | One complete pass through the training data |
| **LLM** | Large Language Model (GPT-4, Llama, etc.) |
| **API** | Application Programming Interface - how programs talk to each other |
| **ORM** | Object-Relational Mapping - Python objects ↔ database rows |
| **Signal/Slot** | Qt's event system for UI communication |

---

## Questions?

This guide covers the essential concepts. For specific questions:
1. Read the relevant source file
2. Check the inline comments
3. Experiment with small changes
4. Use the chat agent to explore: "What can you do?"

