# 🔭 AstroLens Discovery Guide

Your step-by-step guide to finding astronomical discoveries with AstroLens.

---

## 🎯 The Goal

AstroLens isn't just an image viewer—it's a **discovery engine**. The goal is to:
1. Analyze thousands of astronomical images automatically
2. Flag unusual patterns the model hasn't seen before
3. Help you find something no one else has noticed

---

## 📋 Quick Start Checklist

```
□ Step 1: Download training datasets
□ Step 2: Fine-tune the model  
□ Step 3: Start the app
□ Step 4: Download fresh sky images
□ Step 5: Analyze and review anomalies
□ Step 6: Add discoveries to training data
□ Step 7: Periodically retrain
```

---

## Step 1: Download Training Datasets

The model needs to learn what "normal" looks like before it can find anomalies.

### Terminal Commands

```bash
# Activate your environment
cd astroLens
source .venv/bin/activate

# Download Galaxy10 (essential - 17k galaxy images)
python finetuning/download_datasets.py --dataset galaxy10

# Download anomaly samples (teaches unusual patterns)
python finetuning/download_datasets.py --dataset anomalies

# Or download everything at once
python finetuning/download_datasets.py --all
```

### What Each Dataset Teaches

| Dataset | Images | What it teaches |
|---------|--------|-----------------|
| **galaxy10** | 17,736 | 10 common galaxy shapes (spiral, elliptical, etc.) |
| **galaxy_zoo** | ~2,000 | Deeper morphological variations |
| **anomalies** | ~400 | Unusual objects (supernovae candidates, artifacts) |
| **custom** | You add | Your verified discoveries |

---

## Step 2: Fine-Tune the Model

Training makes the model smarter at recognizing patterns.

### Basic Training (5-10 minutes)

```bash
# Train on Galaxy10 first (essential)
python finetuning/train.py --data_dir finetuning/datasets/galaxy10 --epochs 5

# Then add anomaly awareness
python finetuning/train.py --data_dir finetuning/datasets/anomalies --epochs 3
```

### Better Training (30-60 minutes)

```bash
# More epochs = better accuracy
python finetuning/train.py --data_dir finetuning/datasets/galaxy10 --epochs 15 --batch_size 16
```

### Understanding the Settings

- **Epochs**: Number of passes through the data. More = better, but slower.
  - 5 epochs: Quick, decent results
  - 10 epochs: Good balance
  - 20 epochs: High accuracy

- **Batch Size**: Images processed together. 
  - 16: Default, works on most machines
  - 8: If you get memory errors
  - 32: If you have lots of RAM

---

## Step 3: Start the App

```bash
# Terminal 1: Start the API
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start the UI
python ui/main.py
```

---

## Step 4: Download Fresh Sky Images

Get new images to search for discoveries.

### Option A: Via the App

1. Open AstroLens
2. Go to **Control Center** (sidebar)
3. Click **Data Sources** tab
4. Enable sources (SDSS recommended)
5. Set count (e.g., 100)
6. Click **Start Download**

### Option B: Via Command Line

```bash
# Download 100 SDSS galaxy images
python scripts/nightly_ingest.py --source sdss --count 100 --upload --analyze

# Daily discovery script (recommended)
python scripts/daily_discovery.py --sources sdss --count 50
```

### Best Sources

| Source | What you get | Update frequency |
|--------|--------------|------------------|
| **SDSS** | Deep galaxy images | Static archive |
| **ZTF** | Transient events (supernovae) | Nightly |
| **APOD** | NASA's best daily image | Daily |

---

## Step 5: Analyze and Review Anomalies

### Automatic Analysis

If you enabled "Auto-analyze" during download, images are analyzed automatically.

### Manual Batch Analysis

1. Go to **Control Center** → **Batch Analysis**
2. Click **Analyze New** (processes unanalyzed images)
3. Watch for anomalies in the "Discoveries" section

### Understanding Anomaly Scores

- **Low score (0-2)**: Normal, matches training data well
- **Medium score (2-5)**: Slightly unusual, worth a glance
- **High score (5+)**: Significantly different from training data—investigate!

---

## Step 6: Add Discoveries to Training Data

When you find something interesting:

### Save Interesting Finds

1. Click on an anomaly in the gallery
2. Click **Annotate** to get AI description
3. If it's genuinely interesting:
   - Right-click → Save image
   - Put in `finetuning/datasets/custom/train/confirmed_discovery/`

### Handle False Positives

If the model flagged something that's actually normal:
- Save to `finetuning/datasets/custom/train/false_positive/`
- This teaches the model what NOT to flag

### Folder Structure

```
finetuning/datasets/custom/
├── train/
│   ├── normal_galaxy/           ← Boring, typical galaxies
│   ├── interesting_anomaly/     ← Unusual but unverified
│   ├── false_positive/          ← Incorrectly flagged as anomaly
│   └── confirmed_discovery/     ← Verified interesting finds
└── test/
    └── (same structure)
```

---

## Step 7: Periodic Retraining

### When to Retrain

- **Weekly**: If you're actively using the tool
- **After 50+ new images**: In your custom dataset
- **After major discoveries**: To update the model's knowledge

### Retraining Command

```bash
# Retrain on your custom discoveries
python finetuning/train.py --data_dir finetuning/datasets/custom --epochs 5

# Full retraining on all data
python finetuning/train.py --data_dir finetuning/datasets/galaxy10 --epochs 10
python finetuning/train.py --data_dir finetuning/datasets/anomalies --epochs 5
python finetuning/train.py --data_dir finetuning/datasets/custom --epochs 5
```

---

## 🔄 Daily Workflow Summary

```
Morning (5 min):
├── Start API and UI
├── Run: python scripts/daily_discovery.py --sources sdss --count 50
└── Check for anomalies

When anomalies appear (2-5 min each):
├── Click to view full image
├── Get AI annotation
├── Save to appropriate folder if interesting
└── Note in personal log

Weekly (30 min):
├── Review saved discoveries
├── Retrain model if you have 20+ new images
└── Rebuild embeddings for "Find Similar"
```

---

## 🏆 Success Metrics

You're using AstroLens well if:

- [ ] Model is fine-tuned (not just pre-trained)
- [ ] You download 50+ new images weekly
- [ ] You review flagged anomalies within 24 hours
- [ ] You save interesting finds to the custom dataset
- [ ] You retrain at least monthly

---

## ❓ Troubleshooting

### "No anomalies found"

1. Lower the threshold: Edit `inference/ood.py`, change `threshold` from 3.0 to 2.0
2. Train on more diverse data: Download and train on `anomalies` dataset
3. Check your images: Very homogeneous data = fewer anomalies

### "Too many false positives"

1. Raise the threshold: Change to 4.0 or 5.0
2. Add false positives to training: Save them to `custom/train/false_positive/`
3. Retrain the model

### "Model takes forever to train"

1. Reduce epochs: Use 3-5 instead of 10+
2. Reduce batch size: Try 8 instead of 16
3. Use smaller dataset: Sample your data

---

## 🚀 Advanced: Automation

### Daily Cron Job (Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add this line (runs at 3 AM daily)
0 3 * * * cd /path/to/astroLens && source .venv/bin/activate && python scripts/daily_discovery.py --sources sdss --count 100 --quiet
```

### Check Results

```bash
# View latest discovery results
cat downloads/*/discovery_results.json | tail -20
```

---

Good luck with your discoveries! 🔭✨

