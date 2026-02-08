# 🔭 AstroLens

<p align="center">
  <img src="assets/logo.png" alt="AstroLens Logo" width="200"/>
</p>

<p align="center">
  <strong>AI-Powered Galaxy Anomaly Discovery System</strong><br>
  <em>Find unusual galaxies hiding in plain sight</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#web-interface">Web Interface</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#building">Building</a>
</p>

---

## What is AstroLens?

AstroLens is an **AI-powered galaxy anomaly discovery system** that uses advanced machine learning to find unusual galaxy morphologies - mergers, tidal features, irregular structures, and potentially new types of objects.

### Galaxy Mode (Current Focus)
- **Vision Transformer + OOD Detection**: Find galaxies that don't fit normal patterns
- **Morphology Analysis**: CAS + Gini-M20 (Concentration, Asymmetry, Smoothness, merger detection)
- **Reconstruction-based Detection**: PCA reconstruction error for anomaly detection
- **Catalog Cross-Reference**: Auto-queries SIMBAD, NED, VizieR
- **Export Results**: CSV, JSON, HTML reports, VOTable for astronomical tools
- **GPU Acceleration**: CUDA, Apple MPS, CPU auto-detection
- **Human-in-the-Loop**: Review and verify anomalies

### Transient Mode
- YOLO-based transient detection (supernovae, novae)
- 3-phase pipeline: Data Collection → Training → Integration
- Real-time transient alerts

### Web Interface
- Browser-based dashboard for Galaxy and Transient modes
- Image gallery with click-to-analyze
- Export results in multiple formats
- Responsive design

## What Makes AstroLens Unique?

| Feature | AstroLens | Traditional Tools |
|---------|-----------|-------------------|
| **Autonomous Discovery** | Runs continuously, discovers anomalies while you sleep | Manual image-by-image review |
| **Self-Improving AI** | Fine-tunes on discoveries, gets smarter over time | Static models |
| **Multi-Source Ingestion** | SDSS, DECaLS, Pan-STARRS, Galaxy Zoo, ZTF | Single source |
| **Catalog Cross-Reference** | Auto-queries SIMBAD, NED, VizieR | Manual lookup |
| **Dual Detection** | ViT+OOD for galaxies, YOLO for transients | Single model |
| **Morphology Analysis** | CAS, Gini-M20, ellipticity, reconstruction | Visual only |
| **Export to Tools** | CSV, JSON, HTML, VOTable (TOPCAT, Aladin) | No export |
| **Web + Desktop** | Both browser-based and native desktop UI | Single interface |
| **GPU Accelerated** | CUDA / Apple MPS / CPU auto-detection | CPU only |

## Features

### 🔍 Autonomous Discovery Loop
- Continuously downloads from multiple astronomical sources
- Analyzes every image with state-of-the-art Vision Transformer
- Tracks anomalies, near-misses, and uncertain detections
- Runs in the background with system notifications

### 🧠 Self-Improving Model
- Fine-tunes on Galaxy Zoo, Galaxy10, and discovered anomalies
- Tracks accuracy improvements over training runs
- Model accuracy: **83.9%** with **+4.9% improvement** from fine-tuning

### 🌌 Multi-Catalog Cross-Reference
- Queries **SIMBAD**, **NED**, and **VizieR** (SDSS DR12)
- Identifies if detections are known objects or potential discoveries
- Human verification workflow for true/false positive labeling

### 📊 Advanced OOD Detection
- **Ensemble voting** with 3 methods: MSP, Energy, Mahalanobis
- **Reconstruction-based**: PCA reconstruction error for feature-space anomalies
- Auto-calibration for optimal thresholds
- Aggressive mode for maximizing discovery rate

### 🌀 Galaxy Morphology Analysis
- **CAS Parameters**: Concentration, Asymmetry, Smoothness
- **Gini-M20 Coefficients**: Merger and interaction detection
- **Ellipticity**: Shape measurement for compact vs diffuse objects
- Automatic classification: irregular, merger, compact

### 🔬 Transient Detection (YOLO)
- YOLOv8 object detection for transient events
- 3-phase automated pipeline: data collection, training, integration
- TNS and ZTF data sources for training

### 📤 Export Results
- **CSV**: For spreadsheets and data analysis
- **JSON**: For programmatic access with full metadata
- **HTML**: Shareable visual reports
- **VOTable**: For TOPCAT, Aladin, DS9 astronomical tools

### 🖥️ Premium Desktop + Web Interface
- Modern dark theme desktop app (PyQt5)
- Browser-based web interface (FastAPI + Jinja2)
- Real-time discovery statistics
- Image gallery with analysis

## Screenshots

<p align="center">
  <img src="screenshots/Screenshot_1.png" alt="Discovery Panel" width="800"/>
  <br><em>Discovery Panel - Real-time anomaly detection</em>
</p>

<p align="center">
  <img src="screenshots/Screenshot_2.png" alt="Verification Panel" width="800"/>
  <br><em>Verification Panel - Cross-reference against astronomical catalogs</em>
</p>

## Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended
- GPU optional but accelerates inference (NVIDIA CUDA or Apple Silicon MPS)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/astrolens.git
cd astrolens

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained weights
python scripts/download_weights.py
```

### Running AstroLens

**Desktop App:**
```bash
# Start the API server
uvicorn api.main:app --port 8000

# In another terminal, launch the desktop app
python -m ui.main
```

**Web Interface:**
```bash
# Start the API server
uvicorn api.main:app --port 8000

# In another terminal, start the web UI
python -m web.app --port 8080
# Open http://localhost:8080 in your browser
```

**Autonomous Discovery:**
```bash
python scripts/discovery_loop.py
```

**Docker:**
```bash
docker-compose up -d
```

## Web Interface

The web interface provides a browser-based alternative to the desktop app:

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Overview with stats and quick actions |
| Galaxy Mode | `/galaxy` | Browse and analyze galaxy images |
| Transient Mode | `/transient` | YOLO pipeline status and controls |
| Verify | `/verify` | Cross-reference results |
| Export | `/export` | Export results in multiple formats |

## Data Sources

AstroLens downloads galaxy images from multiple astronomical archives:

| Source | Coverage | Depth | Status |
|--------|----------|-------|--------|
| **DECaLS** (DESI Legacy) | 14,000 sq deg | Deep (g,r,z) | ✓ Active |
| **SDSS** (DR18) | 14,500 sq deg | Medium (ugriz) | ✓ Active |
| **Pan-STARRS** (PS1) | 30,000 sq deg | Medium (grizy) | ✓ Active |
| **Hubble Legacy** | Pointed | Very deep | ✓ Active |
| **Galaxy Zoo** | SDSS footprint | Citizen science | ✓ Active |

## Building Executables

Build standalone executables for distribution:

```bash
# Install PyInstaller
pip install pyinstaller

# Build for current platform
python build/build.py

# Generate spec file only
python build/build.py --spec-only

# Clean build artifacts
python build/build.py --clean
```

**Output:**
- macOS: `build/dist/macos/AstroLens.app`
- Linux: `build/dist/linux/AstroLens`
- Windows: `build/dist/windows/AstroLens.exe`

## Architecture

```
astrolens/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── models.py          # Pydantic schemas
│   └── db.py              # SQLite database
├── inference/              # AI inference
│   ├── classifier.py      # ViT-based classifier
│   ├── ood.py             # OOD ensemble detection
│   ├── yolo_detector.py   # YOLO transient detector
│   ├── gpu_utils.py       # GPU acceleration utility
│   └── embeddings.py      # FAISS similarity search
├── features/               # Feature extraction
│   ├── morphology.py      # CAS, Gini-M20, ellipticity
│   ├── reconstruction.py  # PCA reconstruction anomaly
│   ├── export.py          # CSV/JSON/HTML/VOTable export
│   ├── time_series.py     # Light curve features
│   └── multiband.py       # Color analysis
├── catalog/                # Astronomical catalogs
│   └── cross_reference.py # SIMBAD, NED, VizieR
├── transient_detector/     # YOLO transient pipeline
│   ├── pipeline.py        # 3-phase pipeline
│   ├── data_collector.py  # TNS/ZTF data download
│   └── trainer.py         # YOLO training
├── scripts/                # Automation
│   ├── discovery_loop.py  # Autonomous discovery
│   ├── data_sources.py    # Multi-source download
│   └── batch_analyze.py   # Batch OOD analysis
├── ui/                     # Desktop interface (PyQt5)
│   ├── main.py            # App entry point
│   ├── main_window.py     # Main window
│   ├── discovery_panel.py # Discovery controls
│   ├── verification_panel.py # Verification & morphology
│   ├── transient_panel.py # Transient pipeline UI
│   └── gallery.py         # Image gallery
├── web/                    # Web interface (FastAPI)
│   ├── app.py             # Web server
│   └── templates/         # Jinja2 templates
├── build/                  # Build pipeline
│   └── build.py           # PyInstaller build script
└── tests/                  # Test suite
    ├── test_all_features.py    # Comprehensive tests
    ├── test_ui_components.py   # UI tests
    ├── test_morphology.py      # Morphology unit tests
    └── test_data_sources.py    # Data source verification
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with ML status |
| `/stats` | GET | Database statistics |
| `/images` | GET | List all images |
| `/images/{id}` | GET | Get image details |
| `/images/{id}/file` | GET | Get image file |
| `/candidates` | GET | List anomaly candidates |
| `/analysis/full/{id}` | POST | Full analysis pipeline |
| `/analysis/classify/{id}` | POST | Classify image |
| `/analysis/anomaly/{id}` | POST | OOD anomaly detection |
| `/crossref/{id}` | POST | Cross-reference catalogs |
| `/crossref/summary` | GET | Cross-reference stats |
| `/crossref/batch` | POST | Batch cross-reference |

## Model Performance

| Metric | Value |
|--------|-------|
| **Galaxy Classification Accuracy** | 83.9% |
| **Training Improvement** | +4.9% |
| **OOD Detection Methods** | MSP, Energy, Mahalanobis, PCA Reconstruction |
| **Morphology Features** | CAS, Gini-M20, Ellipticity |
| **Catalog Sources** | SIMBAD, NED, VizieR/SDSS |
| **Data Sources** | DECaLS, SDSS, Pan-STARRS, Hubble |
| **Inference Time** | ~274ms per image (CPU), ~50ms (GPU) |

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for Vision Transformer models
- [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/) for citizen science data
- [SIMBAD](http://simbad.u-strasbg.fr/) for astronomical database access
- [NED](https://ned.ipac.caltech.edu/) for extragalactic data
- [SDSS](https://www.sdss.org/) for galaxy survey data
- [DECaLS](https://www.legacysurvey.org/) for DESI Legacy Survey
- [Pan-STARRS](https://panstarrs.stsci.edu/) for wide-field imaging

## Author

**Saman Tabatabaeian**

- Email: saman.tabatabaeian@gmail.com
- LinkedIn: [linkedin.com/in/samantabatabaeian](https://www.linkedin.com/in/samantabatabaeian/)
