# AstroLens Testing Guide

Step-by-step instructions to test every feature of AstroLens.

---

## Prerequisites

```bash
cd astroLens
source .venv/bin/activate
```

If you don't have the virtual environment yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 1: Start the API Backend

Open a terminal and run:

```bash
uvicorn api.main:app --port 8000
```

**What to verify:**
- You see `Uvicorn running on http://0.0.0.0:8000`
- Open http://localhost:8000/health in your browser -- should return `{"status":"ok","version":"1.0.0",...}`
- Open http://localhost:8000/stats -- should show image/anomaly counts
- Open http://localhost:8000/device -- should show your GPU info (e.g., `"device_type":"mps","device_name":"Apple M4"`)

---

## Step 2: Test the Desktop App (Galaxy Mode)

Open a **second terminal** and run:

```bash
python -m ui.main
```

**What to verify:**

### 2a. Gallery
- The main window opens with a dark theme
- Gallery shows images (if any exist in the database)
- Clicking an image opens the Viewer with classification details
- "Upload Images" button works

### 2b. Discovery Loop
- Click "Discovery Loop" in the sidebar
- Click "Batch Analyze" to analyze existing images
- The log should show OOD scores and anomaly counts
- Threshold is auto-calibrated (90th percentile)

### 2c. Verify Panel
- Go to Settings > Verify tab
- Click "Calibrate OOD" -- should show score statistics and recommended threshold
- Click "Analyze Morphology" -- should compute CAS, Gini-M20 for each anomaly
- Click "Cross-Reference All" -- queries SIMBAD/NED for each anomaly
- Check that results appear in the results list (Known vs Unknown)
- Click on an image name -- it should open the file on disk

### 2d. Transient Pipeline
- Go to Settings > Transient tab
- Click "Start Pipeline" to begin data collection
- Watch the phase cards update with progress
- Phase 1: Data collection (TNS + ZTF downloads)
- Phase 2: YOLO training (100 epochs)
- Phase 3: Integration
- Click "Stop" to halt, "Reset" to clear state

---

## Step 3: Test the Web Interface

Open a **third terminal** and run:

```bash
python -m web.app --port 8080
```

Then open http://localhost:8080 in your browser.

**What to verify:**

### 3a. Dashboard (/)
- Shows stats cards (Total Images, Anomalies, Analyzed)
- Links to Galaxy Mode and Transient Mode work
- Recent anomaly candidates table shows data

### 3b. Galaxy Mode (/galaxy)
- Image grid loads with thumbnails
- Clicking an image triggers analysis (classification + OOD + similarity)
- Pagination (Next/Previous) works
- Stats cards show correct counts

### 3c. Transient Mode (/transient)
- Shows pipeline phase cards with progress
- Auto-refreshes if pipeline is running
- Describes each phase clearly

### 3d. Verify (/verify)
- Shows cross-reference results table
- Known/Unknown badges are correctly colored
- Stats cards reflect data

### 3e. Export (/export)
- Click "Export CSV" -- see success message with file path
- Click "Export JSON" -- see success message
- Click "Export HTML Report" -- see success message
- Click "Export VOTable" -- see success message
- All exports saved to `../astrolens_artifacts/exports/`

---

## Step 4: Test Data Sources

```bash
python scripts/data_sources.py --test
```

**What to verify:**
- DECaLS: Connected, image downloaded
- SDSS: Connected, image downloaded
- Pan-STARRS: Connected, image downloaded
- Downloads saved to `../astrolens_artifacts/data/downloads/`

Download known galaxies:

```bash
python scripts/data_sources.py --known
```

Should download M31, M51, M81, etc.

---

## Step 5: Test GPU Detection

```bash
python -c "from inference.gpu_utils import get_device_summary; print(get_device_summary())"
```

**Expected output:**
- macOS with Apple Silicon: `GPU: Apple M4 (Metal Performance Shaders)`
- NVIDIA GPU: `GPU: NVIDIA GeForce RTX ... (X.X GB, CUDA 12.x)`
- No GPU: `CPU: CPU (arm) (No GPU acceleration)`

---

## Step 6: Run Automated Test Suite

### All local tests (no API needed):

```bash
python tests/test_ui_components.py
```

Expected: 14/14 passed.

### Feature tests by section (API must be running):

```bash
python tests/test_all_features.py --section ood
python tests/test_all_features.py --section morphology
python tests/test_all_features.py --section reconstruction
python tests/test_all_features.py --section gpu
python tests/test_all_features.py --section web
python tests/test_all_features.py --section build
```

### Full test suite (API must be running):

```bash
python tests/test_all_features.py
```

### Morphology unit tests:

```bash
python -m pytest tests/test_morphology.py -v
```

**Check logs at:** `../astrolens_artifacts/test_logs/`

---

## Step 7: Test Export Formats

With the API running:

```bash
python -c "
from features.export import ResultsExporter
e = ResultsExporter()
print('CSV:', e.export_csv())
print('JSON:', e.export_json())
print('HTML:', e.export_html())
print('VOTable:', e.export_votable())
e.close()
"
```

Open the HTML report in a browser to verify it looks correct.

---

## Step 8: Build Executable

```bash
pip install pyinstaller
python build/build.py
```

**What to verify:**
- Spec file generated in `build/specs/`
- Build output in `build/dist/{platform}/AstroLens`
- On macOS: `AstroLens.app` bundle is created
- Launch the built app to verify it runs standalone

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API not connecting | Make sure `uvicorn api.main:app --port 8000` is running |
| No images in gallery | Upload images or run discovery loop first |
| No anomalies | Click "Batch Analyze" in Discovery panel |
| MPS not detected | Requires macOS 14+ and Apple Silicon. Check with `python -c "import torch; print(torch.backends.mps.is_built())"` |
| YOLO model not found | Copy from `models/yolo_transient_v1.pt` or run Transient pipeline |
| Cross-reference fails | Check internet connection (queries SIMBAD/NED servers) |
| PyInstaller fails | Ensure `pip install pyinstaller` and no antivirus blocking |
