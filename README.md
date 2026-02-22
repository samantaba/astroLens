<p align="center">
  <img src="assets/logo.png" alt="AstroLens Logo" width="180"/>
</p>

<h1 align="center">AstroLens</h1>

<p align="center">
  <strong>AI that watches the sky and finds what others miss.</strong>
</p>

<p align="center">
  <a href="https://github.com/samantaba/astroLens/stargazers"><img src="https://img.shields.io/github/stars/samantaba/astroLens?style=social" alt="Stars"></a>
  <img src="https://img.shields.io/badge/version-1.1.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-orange" alt="GPU Support">
</p>

<p align="center">
  <a href="#what-is-astrolens">What Is It</a> &bull;
  <a href="#what-it-found">What It Found</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#streaming-discovery-v110">Streaming Discovery</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="https://github.com/samantaba/astroLens/wiki">Wiki</a> &bull;
  <a href="https://deepfieldlabs.github.io">Website</a>
</p>

---

## What Is AstroLens?

AstroLens is an open-source tool that automatically scans the sky for unusual astronomical objects. You start it, and it does the rest: it downloads real images from major sky surveys, analyzes every one of them using AI, and tells you if it finds something interesting.

Think of it as a tireless research assistant that never sleeps. It watches the sky 24/7, learns as it goes, corrects its own mistakes, and presents you with a shortlist of the most promising discoveries -- all without any human supervision.

**In autonomous streaming discovery, AstroLens:**

- Analyzed **22,195 images** from 7 astronomical survey sources across **5,471 unique sky regions**
- Identified **3,541 anomaly candidates** and cross-referenced **269 known objects** against SIMBAD and NED catalogs
- Independently recovered notable objects including **Supernova SN 2014J**, galaxy merger **NGC 3690**, and gravitational lens **SDSS J0252+0039** -- without being told what to look for
- Trained its own transient detection model to **99.5% accuracy** using data it collected during the run
- Made **146 self-correction adjustments** with zero errors and zero human intervention

---

## What It Found

During streaming discovery, AstroLens independently flagged these known, published objects -- validating that the detection pipeline works on real science:

| Object | What It Is | YOLO Confidence | OOD Score |
|--------|-----------|-----------------|-----------|
| **SN 2014J** | Type Ia supernova in galaxy M82, 11.4M light-years away | 52% | 0.26 |
| **NGC 3690** | Violent galaxy merger (Arp 299), one of the most luminous infrared galaxies nearby | 60% | 0.11 |
| **SDSS J0252+0039** | Confirmed strong gravitational lens -- massive galaxy bending background light | 61% | 0.35 |

All detections were cross-referenced against the **SIMBAD** and **NED** astronomical databases. In total, **269 known objects** were independently recovered and confirmed, with **173 having published references**. The system found these objects on its own, using only the AI models and no prior knowledge of what was in the images.

---

## Who Is This For?

- **Astronomy enthusiasts** who want to discover things for themselves
- **Researchers** who need an automated anomaly detection pipeline for survey data
- **ML engineers** interested in a production Vision Transformer + OOD + YOLO system
- **Anyone** curious about what happens when you point AI at the sky and let it run

---

## Features

### Galaxy Mode (v1.0)

| Feature | Description |
|---------|-------------|
| **ViT + OOD Ensemble** | Classifies galaxies with a Vision Transformer, then flags anomalies using 3 OOD methods (MSP, Energy, Mahalanobis) with majority voting |
| **Galaxy Morphology** | Computes Concentration, Asymmetry, Smoothness (CAS), Gini-M20 coefficients, and Ellipticity to characterize galaxy structure |
| **PCA Reconstruction** | Detects anomalies by measuring how poorly an image reconstructs from learned principal components |
| **Catalog Cross-Reference** | Automatically queries SIMBAD, NED, and VizieR to determine if a detection is a known object or potentially new |
| **Adaptive Threshold** | Auto-calibrates OOD threshold to the 90th percentile of existing scores -- no manual tuning needed |
| **Human Verification** | Mark results as True Positive / False Positive to improve future accuracy |
| **Export** | Save results as CSV, JSON, HTML reports, or VOTable (for TOPCAT, Aladin, DS9) |

### Transient Mode (v1.0)

| Feature | Description |
|---------|-------------|
| **YOLO Detection** | YOLOv8 object detection trained on real transient images from TNS and ZTF |
| **3-Phase Pipeline** | Automated: (1) Data collection, (2) Model training, (3) Integration with discovery |
| **Pre-trained Model** | Ships with a trained model so users can detect transients immediately |

### Streaming Discovery (v1.1.0)

| Feature | Description |
|---------|-------------|
| **Multi-Day Streaming** | Runs autonomously for days, downloading and analyzing images 24/7 |
| **Self-Correcting Intelligence** | Auto-adjusts thresholds, rebalances data sources, improves as it runs |
| **Daily Reports** | Generates HTML reports with charts, candidate rankings, and trend analysis |
| **Publishing Pipeline** | Surfaces the best candidates with supporting evidence for publication |
| **UI + CLI** | Start, monitor, and stop streaming from the desktop app or command line |

### Infrastructure

| Feature | Description |
|---------|-------------|
| **Web Interface** | Browser-based UI for Galaxy Mode, Transient Mode, Verification, and Export |
| **Desktop App** | Native PyQt5 application with premium dark theme |
| **GPU Acceleration** | Auto-detects NVIDIA CUDA, Apple MPS, or falls back to CPU |
| **Multi-Source Data** | Downloads from DECaLS, SDSS, Pan-STARRS, Hubble, Galaxy Zoo |
| **CI/CD** | GitHub Actions pipelines build executables for macOS, Linux, and Windows on every release |

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/samantaba/astroLens.git
cd astroLens
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start API

```bash
uvicorn api.main:app --port 8000
```

### 3. Launch UI

**Desktop app:**
```bash
python -m ui.main
```

**Or web interface:**
```bash
python -m web.app --port 8080
# Open http://localhost:8080
```

---

## Demo

<p align="center">
  <img src="assets/detect.gif" alt="Anomaly Detection" width="800"/>
  <br><em>Anomaly Detection -- AI identifies unusual objects with OOD scoring and YOLO transient detection</em>
</p>

<p align="center">
  <img src="assets/viewer.gif" alt="Gallery and Viewer" width="800"/>
  <br><em>Gallery and Viewer -- Browse, inspect, and verify detected anomaly candidates</em>
</p>

<p align="center">
  <img src="assets/visualise.gif" alt="Web Analysis" width="800"/>
  <br><em>Web Dashboard -- Statistics, streaming charts, and analysis insights in the browser</em>
</p>

## Screenshots

<p align="center">
  <img src="screenshots/Screenshot_1.png" alt="Discovery Panel" width="800"/>
  <br><em>Discovery Panel -- Real-time anomaly detection with OOD ensemble</em>
</p>

<p align="center">
  <img src="screenshots/Screenshot_2.png" alt="Verification Panel" width="800"/>
  <br><em>Verification Panel -- Cross-reference against SIMBAD, NED, VizieR</em>
</p>

<p align="center">
  <img src="screenshots/Screenshot_3.png" alt="Gallery" width="800"/>
  <br><em>Image Gallery -- Browse and inspect detected anomalies</em>
</p>

---

## Web Interface

The web interface runs at `http://localhost:8080` and provides:

| Page | Path | What It Does |
|------|------|-------------|
| Dashboard | `/` | Overview stats, quick links to Galaxy and Transient modes |
| Galaxy Mode | `/galaxy` | Browse images, click to run full analysis pipeline |
| Transient Mode | `/transient` | Monitor YOLO training pipeline progress |
| Verify | `/verify` | View cross-reference results (Known vs Unknown) |
| Export | `/export` | One-click export to CSV, JSON, HTML, VOTable |
| Streaming | `/streaming` | Live streaming discovery dashboard with charts |

---

## How It Works

```
 Source Images           Vision Transformer          OOD Ensemble
(SDSS, DECaLS, etc.)        (ViT)              (MSP + Energy + Mahal.)
       |                      |                        |
       v                      v                        v
  +---------+          +------------+           +------------+
  | Download| -------> | Classify   | --------> | Anomaly?   |
  +---------+          +------------+           +-----+------+
                                                      |
                                          +-----------+-----------+
                                          |                       |
                                    Yes (2+ votes)           No (normal)
                                          |                       |
                                          v                       v
                                   +-----------+           Track for
                                   | Morphology|           fine-tuning
                                   | CAS/Gini  |
                                   +-----+-----+
                                         |
                                         v
                                  +-------------+
                                  | Cross-Ref   |
                                  | SIMBAD/NED  |
                                  +------+------+
                                         |
                                  +------+------+
                                  |             |
                               Known        Unknown
                               Object    (Potential Discovery)
```

---

## Data Sources

| Source | Coverage | Bands | Type |
|--------|----------|-------|------|
| DECaLS (DESI Legacy) | 14,000 sq deg | g, r, z | Deep imaging |
| SDSS (DR18) | 14,500 sq deg | u, g, r, i, z | Classic survey |
| Pan-STARRS (PS1) | 30,000 sq deg | g, r, i, z, y | Wide-field |
| Hubble Legacy | Pointed fields | Multi-band | High resolution |
| Galaxy Zoo | SDSS footprint | -- | Citizen science labels |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Galaxy Classification Accuracy | 83.9% |
| Fine-tuning Improvement | +4.9% |
| OOD Detection Methods | MSP, Energy, Mahalanobis (ensemble) |
| Morphology Features | CAS, Gini-M20, Ellipticity |
| YOLO Transient mAP50 | 99.5% (v1.1 -- trained on 2,742 images / 5,234 annotations) |
| Inference Time (CPU) | ~274ms/image |
| Inference Time (MPS) | ~50ms/image |

---

## Milestone: v1.1.0

Streaming discovery with self-correcting intelligence. Validated in autonomous operation: 22,195 images analyzed across 5,471 sky regions, 3,541 anomaly candidates identified, 269 known astronomical objects independently recovered and confirmed against SIMBAD/NED.

**ML Pipeline**
- Vision Transformer fine-tuned on Galaxy10 + Galaxy Zoo (83.9% accuracy, +4.9% improvement)
- Ensemble OOD detection with auto-calibrating thresholds
- PCA reconstruction-based anomaly detection
- YOLOv8 transient detector trained on TNS + ZTF data

**Analysis Features**
- Galaxy morphology: CAS parameters, Gini-M20, ellipticity
- Catalog cross-reference: SIMBAD, NED, VizieR (SDSS DR12)
- Adaptive threshold calibration (90th percentile)
- Result export in 4 formats (CSV, JSON, HTML, VOTable)

**Infrastructure**
- Desktop app (PyQt5) with premium dark UI
- Web interface (FastAPI + Jinja2) for browser access
- GPU auto-detection (CUDA / Apple MPS / CPU)
- Multi-source data pipeline (DECaLS, SDSS, Pan-STARRS, Hubble)
- CI/CD: GitHub Actions build for macOS, Linux, Windows
- Pre-trained YOLO models included (v1.0 + v1.1 fine-tuned)

---

## Building Executables

```bash
pip install pyinstaller
python build/build.py
```

Or let GitHub Actions build automatically on each tagged release:

```bash
git tag v1.1.0
git push origin v1.1.0
```

This triggers the release workflow, which builds for all 3 platforms and creates a GitHub Release with downloadable executables.

---

## Testing

See [TESTING.md](TESTING.md) for the full step-by-step guide.

Quick automated tests:

```bash
python tests/test_ui_components.py          # UI imports (14 tests)
python tests/test_all_features.py           # Full suite (20+ tests)
python -m pytest tests/test_morphology.py   # Morphology unit tests
python tests/test_data_sources.py           # Data source verification
```

---

## Architecture

```
astroLens/
├── api/                     FastAPI backend (REST API)
├── inference/               ML models (ViT, OOD, YOLO, GPU utils)
├── features/                Feature extraction (morphology, PCA, export)
├── catalog/                 Astronomical catalog queries
├── transient_detector/      YOLO transient pipeline
├── scripts/                 Automation (discovery, data sources, batch)
├── ui/                      Desktop app (PyQt5)
├── web/                     Web interface (FastAPI + Jinja2)
├── build/                   PyInstaller build pipeline
├── models/                  Pre-trained models (Git LFS)
└── tests/                   Test suite with logging
```

---

## Configuration

```bash
# GPU: auto-detected, no config needed
# To force CPU: export ASTROLENS_DEVICE=cpu

# LLM annotation (optional):
export LLM_PROVIDER=ollama    # or "openai"
export OPENAI_API_KEY=sk-...  # if using OpenAI

# NASA APOD images (optional):
export NASA_API_KEY=...
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, ML model status |
| `/device` | GET | GPU/hardware information |
| `/stats` | GET | Database statistics |
| `/images` | GET | List images (pagination, filters) |
| `/images/{id}` | GET | Image details |
| `/images/{id}/file` | GET | Serve image file |
| `/candidates` | GET | Anomaly candidates (sorted by OOD score) |
| `/analysis/full/{id}` | POST | Full pipeline: classify + OOD + embed + similar |
| `/analysis/classify/{id}` | POST | ViT classification only |
| `/analysis/anomaly/{id}` | POST | OOD detection only |
| `/crossref/{id}` | POST | Cross-reference against catalogs |
| `/crossref/summary` | GET | Cross-reference statistics |
| `/crossref/batch` | POST | Batch cross-reference all anomalies |

---

## Streaming Discovery (v1.1.0)

AstroLens can run **autonomously for days**, continuously downloading, analyzing, and reporting anomaly candidates.

```bash
# Start streaming discovery (runs until stopped)
python scripts/streaming_discovery.py --days 7

# Or launch from the UI: Settings > Streaming Discovery > Start
```

What the streaming engine does:
- Downloads images 24/7 from 5+ astronomical surveys
- Runs full ViT + OOD + morphology + catalog pipeline on every image
- **Self-corrects**: adjusts thresholds, rebalances sources, re-prioritizes based on what it finds
- Generates **daily HTML reports** with charts, candidate rankings, and improvement metrics
- Saves everything to local files (never pushes to Git)
- Produces a **publishing-ready summary** of the best candidates

Reports are saved to `astrolens_artifacts/streaming_reports/`.

---

## Contributing

1. Fork the repository
2. Create a branch (`git checkout -b feature/your-feature`)
3. Make changes and test (`python tests/test_all_features.py`)
4. Commit (`git commit -m 'Add your feature'`)
5. Push and open a Pull Request

CI will run tests automatically on your PR.

---

## Support

- **Website:** [deepfieldlabs.github.io](https://deepfieldlabs.github.io)
- **Issues:** [GitHub Issues](https://github.com/samantaba/astroLens/issues)
- **Discussions:** [GitHub Discussions](https://github.com/samantaba/astroLens/discussions)
- **Wiki:** [Full Documentation](https://github.com/samantaba/astroLens/wiki)

---

<p align="center">
  <strong>If AstroLens helps your research or you like the idea, please <a href="https://github.com/samantaba/astroLens/stargazers">give it a star</a>.</strong><br>
  It takes 2 seconds and helps others find the project.
</p>

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Saman Tabatabaeian**

- [LinkedIn](https://www.linkedin.com/in/samantabatabaeian/)
- saman.tabatabaeian@gmail.com
