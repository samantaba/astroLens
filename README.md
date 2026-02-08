<p align="center">
  <img src="assets/logo.png" alt="AstroLens Logo" width="180"/>
</p>

<h1 align="center">AstroLens</h1>

<p align="center">
  <strong>AI-Powered Galaxy Anomaly Discovery System</strong><br>
  <em>Find unusual galaxies that traditional surveys miss</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20MPS%20%7C%20CPU-orange" alt="GPU Support">
</p>

<p align="center">
  <a href="#why-astrolens">Why AstroLens</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#web-interface">Web Interface</a> &bull;
  <a href="#testing">Testing</a> &bull;
  <a href="#building">Building</a> &bull;
  <a href="docs/WIKI.md">Wiki</a>
</p>

---

## Why AstroLens?

Astronomical surveys generate millions of images. Most get classified by standard pipelines and never looked at again. **AstroLens finds what those pipelines miss** -- unusual galaxy morphologies, merger candidates, compact objects, and structures that don't fit known categories.

It does this by combining a **Vision Transformer** (ViT) with an **ensemble of out-of-distribution detectors**, then cross-referencing results against real astronomical catalogs (SIMBAD, NED, VizieR). If an object is unknown to existing catalogs, it could be a genuine new discovery.

**Galaxy Mode** focuses on morphological anomalies. **Transient Mode** uses YOLO to detect supernovae and variable stars. Both run autonomously.

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
| YOLO Transient mAP50 | 51.5% (v1 -- improving with more data) |
| Inference Time (CPU) | ~274ms/image |
| Inference Time (MPS) | ~50ms/image |

---

## Milestone: v1.0.0

This release represents months of development. Here is what was built:

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
- Pre-trained YOLO model included via Git LFS

---

## Building Executables

```bash
pip install pyinstaller
python build/build.py
```

Or let GitHub Actions build automatically on each tagged release:

```bash
git tag v1.0.0
git push origin v1.0.0
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

## Contributing

1. Fork the repository
2. Create a branch (`git checkout -b feature/your-feature`)
3. Make changes and test (`python tests/test_all_features.py`)
4. Commit (`git commit -m 'Add your feature'`)
5. Push and open a Pull Request

CI will run tests automatically on your PR.

---

## Support

- **Issues:** [GitHub Issues](https://github.com/samantaba/astroLens/issues)
- **Discussions:** [GitHub Discussions](https://github.com/samantaba/astroLens/discussions)
- **Wiki:** [Full Documentation](docs/WIKI.md)

If AstroLens helps your research, please consider starring the repository.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Saman Tabatabaeian**

- [LinkedIn](https://www.linkedin.com/in/samantabatabaeian/)
- saman.tabatabaeian@gmail.com
