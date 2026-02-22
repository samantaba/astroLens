# AstroLens Wiki

Comprehensive documentation for AstroLens -- the AI-Powered Galaxy Anomaly Discovery System.

**Website:** [deepfieldlabs.github.io](https://deepfieldlabs.github.io) | **GitHub:** [samantaba/astroLens](https://github.com/samantaba/astroLens)

---

## Table of Contents

1. [Overview](#overview)
2. [Galaxy Mode](#galaxy-mode)
   - [Vision Transformer (ViT)](#vision-transformer)
   - [Out-of-Distribution Detection](#out-of-distribution-detection)
   - [Galaxy Morphology Analysis](#galaxy-morphology-analysis)
   - [PCA Reconstruction Detection](#pca-reconstruction-detection)
   - [Catalog Cross-Reference](#catalog-cross-reference)
3. [Transient Mode](#transient-mode)
   - [YOLO Object Detection](#yolo-object-detection)
   - [Training Pipeline](#training-pipeline)
   - [Data Sources for Training](#transient-data-sources)
4. [Fine-Tuning Methodology](#fine-tuning-methodology)
5. [GPU Acceleration](#gpu-acceleration)
6. [Data Sources](#data-sources)
7. [Export Formats](#export-formats)
8. [Web Interface](#web-interface)
9. [API Reference](#api-reference)
10. [Build and Distribution](#build-and-distribution)
11. [Troubleshooting](#troubleshooting)

---

## Overview

AstroLens uses machine learning to automatically discover unusual astronomical objects in survey images. It operates in two modes:

- **Galaxy Mode**: Finds unusual galaxy morphologies (mergers, irregulars, compact objects) using a Vision Transformer + out-of-distribution detection ensemble.
- **Transient Mode**: Detects transient events (supernovae, variable stars) using YOLO object detection.

Both modes can run autonomously, continuously downloading and analyzing new images.

---

## Galaxy Mode

### Vision Transformer

AstroLens uses a **Vision Transformer (ViT)** as its core image classifier. ViT was originally developed by Google and treats images as sequences of patches, similar to how language models process words.

**How it works in AstroLens:**
1. An image is split into 16x16 pixel patches
2. Each patch is embedded into a vector
3. The transformer processes all patches with self-attention
4. The final output classifies the galaxy into one of 10 types (from Galaxy10 dataset)

**Fine-tuning results:**
- Base model accuracy: 79.0%
- After fine-tuning on Galaxy10 + Galaxy Zoo: **83.9%** (+4.9% improvement)
- Training: 10 epochs, AdamW optimizer, cosine LR schedule

The classifier output (logits) feeds into the OOD detection pipeline.

### Out-of-Distribution Detection

OOD detection identifies images that don't fit the patterns the model learned during training. If a galaxy looks unlike anything in the training data, it might be genuinely unusual.

AstroLens uses **3 complementary methods** with majority voting:

#### 1. Maximum Softmax Probability (MSP)
- Converts classifier logits to probabilities via softmax
- If the highest probability is low (model is uncertain), the image is suspicious
- Score: `1 - max(softmax(logits))`
- Threshold: typically ~0.5 (flag if confidence < 50%)

#### 2. Energy-Based Detection
- Computes a free energy score: `E(x) = -T * log(sum(exp(logit_i / T)))`
- Higher energy = lower confidence = more likely OOD
- More theoretically grounded than MSP
- Based on: Liu et al., NeurIPS 2020

#### 3. Mahalanobis Distance
- Measures distance from the image embedding to the nearest class centroid
- Uses the shared covariance matrix of all classes
- Images far from all centroids are likely OOD
- Requires calibration (computed during training)
- Based on: Lee et al., NeurIPS 2018

**Ensemble voting:** If 2 out of 3 methods flag an image as OOD, it's marked as an anomaly. This reduces false positives compared to using any single method.

**Adaptive threshold:** The threshold auto-calibrates to the 90th percentile of existing OOD scores. The top 10% most unusual images are flagged as anomalies. No manual tuning required.

### Galaxy Morphology Analysis

After OOD detection flags anomaly candidates, morphology analysis characterizes their physical structure.

#### CAS Parameters

**Concentration (C):** Measures how centrally concentrated the light is. Calculated as the ratio of the radii containing 80% and 20% of the total flux.
- High C (>3.5): Elliptical galaxies, compact objects
- Low C (<2.5): Diffuse, irregular galaxies

**Asymmetry (A):** Measures rotational asymmetry. The image is rotated 180 degrees and subtracted from itself. High residuals mean asymmetry.
- High A (>0.35): Merger signatures, tidal features
- Low A (<0.1): Symmetric ellipticals or face-on spirals

**Smoothness (S):** Also called "clumpiness." The image is smoothed with a Gaussian filter and subtracted from the original. High residuals indicate clumpy structure.
- High S (>0.2): Star-forming regions, irregular structure
- Low S (<0.05): Smooth elliptical galaxies

#### Gini-M20 Coefficients

**Gini coefficient:** Measures inequality of pixel brightness distribution. A Gini of 1.0 means all light is in one pixel; 0.0 means perfectly uniform.
- High Gini (>0.55) + Low M20: Concentrated, possible merger

**M20:** The normalized second-order moment of the 20% brightest pixels. Measures how spread out the brightest regions are.
- M20 > -1.0: Light is spread out (irregular or merger)
- M20 < -2.0: Light is concentrated (elliptical)

**Merger detection rule:** Gini > -0.14 * M20 + 0.33 (empirical boundary from Lotz et al. 2004)

#### Ellipticity

Measures how elongated the galaxy appears. Computed from the second-order moments of the image intensity distribution.
- E = 0: Perfectly circular
- E = 1: Maximally elongated

### PCA Reconstruction Detection

An additional anomaly detection method using Principal Component Analysis.

**How it works:**
1. PCA is fit on a sample of "normal" galaxy images, learning the main patterns
2. For a new image, it projects into the PCA space and reconstructs
3. The reconstruction error (MSE between original and reconstruction) is the anomaly score
4. Normal images reconstruct well (low error); anomalies don't (high error)

This complements OOD detection because it works in pixel space rather than feature space, catching different types of anomalies.

### Catalog Cross-Reference

After detection, anomalies are automatically checked against real astronomical databases:

#### SIMBAD (CDS, Strasbourg)
- Contains ~15 million astronomical objects
- Returns object type, identifiers, and bibliographic references
- Uses TAP/ADQL queries via the SIMBAD API

#### NED (NASA/IPAC)
- NASA Extragalactic Database
- ~400 million objects with redshifts and photometry
- Cross-matches by sky position (cone search)

#### VizieR (SDSS DR12)
- Accesses the SDSS PhotoObj catalog
- Returns photometric data (magnitudes in u, g, r, i, z bands)

**Result classification:**
- **Known:** Matched to an existing catalog entry
- **Published:** Has bibliographic references (has been studied)
- **Unknown:** No match found -- potential new discovery

---

## Transient Mode

### YOLO Object Detection

Transient Mode uses **YOLOv8** (You Only Look Once) to detect transient events in images. Unlike the ViT classifier which classifies entire images, YOLO finds and localizes specific objects within images.

**What it detects:**
- Supernovae (stellar explosions)
- Novae (thermonuclear explosions on white dwarfs)
- Variable stars (stars that change brightness)
- Other transient events

**Performance:**
- Trained on real transient images from TNS and ZTF
- mAP50: 51.5% (v1 -- improving with more training data)
- Inference: real-time (~30ms per image on GPU)

### Training Pipeline

The transient pipeline runs in 3 automated phases:

**Phase 1: Data Collection**
- Downloads transient images from the Transient Name Server (TNS)
- Fetches additional images from the Zwicky Transient Facility (ZTF)
- Creates YOLO-format annotations (bounding boxes)
- Validates images and removes corrupted files

**Phase 2: YOLO Training**
- Trains YOLOv8-nano on collected data (100 epochs)
- Uses mosaic augmentation, random flipping, and erasing
- Monitors precision, recall, and mAP during training
- Saves best weights based on validation mAP

**Phase 3: Integration**
- Copies best model to the inference directory
- Tests on existing anomaly candidates
- Reports confirmation rate (what % of OOD anomalies are also transients)

### Transient Data Sources

| Source | Type | Coverage |
|--------|------|----------|
| TNS (Transient Name Server) | Confirmed transients | Global, curated |
| ZTF (Zwicky Transient Facility) | Survey alerts | Northern sky, real-time |

---

## Fine-Tuning Methodology

### Galaxy Classifier (ViT)

**Dataset:** Galaxy10 DECals (17,736 images, 10 classes)

Galaxy10 classes:
1. Disturbed Galaxies
2. Merging Galaxies
3. Round Smooth Galaxies
4. In-between Round Smooth Galaxies
5. Cigar Shaped Smooth Galaxies
6. Barred Spiral Galaxies
7. Unbarred Tight Spiral Galaxies
8. Unbarred Loose Spiral Galaxies
9. Edge-on Galaxies without Bulge
10. Edge-on Galaxies with Bulge

**Training configuration:**
- Base model: `google/vit-base-patch16-224`
- Optimizer: AdamW (lr=5e-5, weight_decay=0.01)
- Schedule: Cosine annealing with warmup (500 steps)
- Epochs: 10
- Batch size: 16
- Image size: 224x224
- Augmentation: Random horizontal flip, rotation, color jitter

**Results:**
- Before fine-tuning: 79.0% accuracy
- After fine-tuning: **83.9% accuracy** (+4.9%)
- Best performance on: Barred spirals (92%), Edge-on (90%)
- Most confused: Disturbed vs Merging galaxies

### YOLO Transient Detector

**Dataset:** TNS + ZTF transient images (~2000 images)

**Training configuration:**
- Model: YOLOv8-nano (lightweight for fast inference)
- Epochs: 100
- Image size: 128x128
- Batch size: 16
- Augmentation: Mosaic, flip, erasing

**Results:**
- Precision: 42.5% (v1)
- Recall: 61.2%
- mAP50: 51.5%
- Expected to improve significantly with more diverse training data

---

## GPU Acceleration

AstroLens automatically detects and uses the best available compute device:

| Device | How It's Detected | Speedup |
|--------|------------------|---------|
| NVIDIA CUDA | `torch.cuda.is_available()` | ~5-10x over CPU |
| Apple MPS | Direct probe fallback (supports macOS 14+) | ~3-5x over CPU |
| CPU | Fallback | Baseline |

**Apple Silicon note:** PyTorch's `mps.is_available()` can return `False` on newer macOS versions due to version detection bugs. AstroLens works around this by directly probing the MPS device.

Check your device:
```bash
python -c "from inference.gpu_utils import get_device_summary; print(get_device_summary())"
```

---

## Data Sources

AstroLens can download galaxy images from multiple surveys:

### DECaLS (DESI Legacy Imaging Survey)
- **Coverage:** ~14,000 sq deg
- **Depth:** 24.0 mag (g), 23.4 mag (r), 22.5 mag (z)
- **Resolution:** 0.262 arcsec/pixel
- **Best for:** Faint galaxies, deep imaging

### SDSS (Sloan Digital Sky Survey, DR18)
- **Coverage:** ~14,500 sq deg (Northern sky)
- **Bands:** u, g, r, i, z
- **Resolution:** 0.4 arcsec/pixel
- **Best for:** Broad coverage, well-characterized photometry

### Pan-STARRS (PS1)
- **Coverage:** ~30,000 sq deg (dec > -30)
- **Bands:** g, r, i, z, y
- **Best for:** Wide-field coverage, 3/4 of the sky

### Hubble Legacy Archive
- **Coverage:** Pointed observations only
- **Resolution:** 0.05 arcsec/pixel (10x better than ground-based)
- **Best for:** High-resolution morphology studies

---

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| CSV | `.csv` | Spreadsheets, pandas, R |
| JSON | `.json` | Programmatic access, includes metadata and cross-ref results |
| HTML | `.html` | Shareable visual report, opens in any browser |
| VOTable | `.vot` | Astronomical tools (TOPCAT, Aladin, DS9, SAOImage) |

---

## Web Interface

The web interface provides browser-based access to all features.

**Start:** `python -m web.app --port 8080`

### Pages

- **Dashboard** (`/`): Stats overview, quick access buttons
- **Galaxy Mode** (`/galaxy`): Image gallery with click-to-analyze
- **Transient Mode** (`/transient`): Pipeline progress monitoring
- **Verify** (`/verify`): Cross-reference results table
- **Export** (`/export`): One-click export in all formats

### API Proxy

The web interface proxies requests to the main API backend (port 8000). Both servers must be running.

---

## API Reference

Full REST API documentation:

### Health and Info
| Endpoint | Method | Response |
|----------|--------|----------|
| `/health` | GET | `{"status":"ok","version":"1.0.0","ml_model_loaded":true}` |
| `/device` | GET | `{"device_type":"mps","device_name":"Apple M4"}` |
| `/stats` | GET | `{"total_images":10000,"anomalies":500,"analyzed":9500}` |

### Images
| Endpoint | Method | Parameters | Response |
|----------|--------|------------|----------|
| `/images` | GET | `skip`, `limit`, `anomaly_only` | List of ImageSummary |
| `/images/{id}` | GET | -- | ImageDetail |
| `/images/{id}/file` | GET | -- | Image file (JPEG/PNG) |
| `/images` | POST | `file` (multipart) | Created ImageSummary |
| `/images/{id}` | PATCH | `class_label`, `ood_score`, `is_anomaly` | Update confirmation |
| `/images/{id}` | DELETE | -- | Deletion confirmation |

### Analysis
| Endpoint | Method | Response |
|----------|--------|----------|
| `/analysis/classify/{id}` | POST | `{"class_label":"Spiral","confidence":0.92}` |
| `/analysis/anomaly/{id}` | POST | `{"ood_score":0.45,"is_anomaly":true,"threshold":0.38}` |
| `/analysis/similar/{id}` | POST | List of similar images with similarity scores |
| `/analysis/full/{id}` | POST | Combined: classification + anomaly + similar |

### Candidates
| Endpoint | Method | Parameters | Response |
|----------|--------|------------|----------|
| `/candidates` | GET | `skip`, `limit` | Anomalies sorted by OOD score |

### Cross-Reference
| Endpoint | Method | Response |
|----------|--------|----------|
| `/crossref/{id}` | POST | `{"is_known":false,"primary_match":null,"status":"unknown"}` |
| `/crossref/summary` | GET | Counts of known/unknown/verified |
| `/crossref/batch` | POST | Batch results for all anomalies |
| `/crossref/{id}/verify` | POST | Mark as true_positive/false_positive |

---

## Build and Distribution

### Building from Source

```bash
pip install pyinstaller
python build/build.py
```

**Output locations:**
- macOS: `build/dist/macos/AstroLens.app`
- Linux: `build/dist/linux/AstroLens/`
- Windows: `build/dist/windows/AstroLens/`

### Automated Releases

Push a version tag to trigger the GitHub Actions release pipeline:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This builds for all 3 platforms and publishes a GitHub Release with:
- `AstroLens-v1.0.0-macos-arm64.zip`
- `AstroLens-v1.0.0-linux-x86_64.tar.gz`
- `AstroLens-v1.0.0-windows-x64.zip`

### Pre-trained Models

The YOLO transient model is included via Git LFS at `models/yolo_transient_v1.pt`. Users don't need to retrain unless they want to customize.

---

## Troubleshooting

### Common Issues

**"API not reachable"**
- Make sure the API is running: `uvicorn api.main:app --port 8000`
- Check http://localhost:8000/health

**"No anomalies found"**
- Run Batch Analyze in the Discovery panel to scan existing images
- The adaptive threshold marks the top 10% as anomalies

**"MPS not available"**
- Requires Apple Silicon (M1/M2/M3/M4) and macOS 14+
- AstroLens probes MPS directly to work around PyTorch version detection bugs
- Verify: `python -c "import torch; t = torch.tensor([1.0], device='mps'); print('MPS works')"`

**"YOLO model not found"**
- The pre-trained model is at `models/yolo_transient_v1.pt`
- If missing, run `git lfs pull` to download it
- Or run the Transient pipeline to train a new one

**"Cross-reference returns no results"**
- Requires internet access (queries SIMBAD/NED servers)
- Images need RA/Dec coordinates in filenames or metadata
- Try increasing the search radius (default: 120 arcsec)

**"PyInstaller build fails"**
- Install: `pip install pyinstaller`
- On macOS, you may need to allow the app in System Preferences > Security
- On Windows, temporarily disable antivirus during build

### Log Files

Test logs are saved to `../astrolens_artifacts/test_logs/` with timestamps. Each test run produces both a `.log` file and a `.json` results summary.

---

*For additional help, open an issue on [GitHub](https://github.com/samantaba/astroLens/issues).*
