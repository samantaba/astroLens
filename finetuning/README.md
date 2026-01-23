# AstroLens Fine-Tuning Guide

This folder contains scripts and documentation for fine-tuning the ViT model on labeled astronomical data.

## Quick Start

```bash
# 1. Download labeled dataset
python finetuning/download_datasets.py --dataset galaxy_zoo

# 2. Fine-tune the model
python finetuning/train.py --epochs 10 --batch_size 32

# 3. Evaluate
python finetuning/evaluate.py --model weights/vit_astrolens
```

---

## Available Datasets

| Dataset | Size | Classes | Source |
|---------|------|---------|--------|
| **Galaxy Zoo 2** | 243,000 images | 11 galaxy types | Kaggle |
| **PLAsTiCC** | 3.5M simulated | Transient types | Kaggle |
| **AstroNN** | 400,000+ spectra | Star types | Hugging Face |
| **Galaxy10 DECals** | 17,736 images | 10 classes | Hugging Face |

---

## File Structure

```
finetuning/
├── README.md           # This file
├── download_datasets.py  # Download labeled datasets
├── train.py            # Fine-tuning script
├── evaluate.py         # Model evaluation
├── config.yaml         # Training configuration
└── datasets/           # Downloaded data (gitignored)
```

---

## Training Process

### 1. Download Data

```bash
# Galaxy Zoo (243K labeled galaxies)
python finetuning/download_datasets.py --dataset galaxy_zoo

# Galaxy10 DECals (smaller, faster)
python finetuning/download_datasets.py --dataset galaxy10
```

### 2. Fine-Tune

```bash
# Basic training
python finetuning/train.py

# Custom settings
python finetuning/train.py \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 2e-5 \
    --output_dir weights/vit_astrolens_v2
```

### 3. Evaluate

```bash
python finetuning/evaluate.py \
    --model weights/vit_astrolens \
    --test_dir finetuning/datasets/galaxy10/test
```

---

## Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  base: "google/vit-base-patch16-224"
  num_labels: 11

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2e-5
  warmup_ratio: 0.1
  weight_decay: 0.01

data:
  train_dir: "finetuning/datasets/galaxy10/train"
  test_dir: "finetuning/datasets/galaxy10/test"
  image_size: 224

output:
  dir: "weights/vit_astrolens"
  save_strategy: "epoch"
```

---

## Expected Results

| Dataset | Epochs | Top-1 Accuracy | Time (GPU) |
|---------|--------|----------------|------------|
| Galaxy10 | 10 | ~92% | ~30 min |
| Galaxy Zoo | 10 | ~87% | ~8 hours |
| Combined | 20 | ~90% | ~12 hours |

---

## Hardware Requirements

| Setup | Training Time | Memory |
|-------|---------------|--------|
| CPU | Very slow (~days) | 8GB+ RAM |
| GPU (RTX 3060) | ~1 hour | 6GB VRAM |
| GPU (A100) | ~10 min | 40GB VRAM |
| Colab (free) | ~2 hours | 12GB VRAM |

---

## Automated Training Pipeline

For automated retraining with new data:

```bash
# Run automated pipeline (downloads → trains → evaluates → saves)
python finetuning/pipeline.py --auto

# Schedule weekly retraining
crontab -e
# Add: 0 2 * * 0 cd /path/to/astroLens && python finetuning/pipeline.py --auto
```

