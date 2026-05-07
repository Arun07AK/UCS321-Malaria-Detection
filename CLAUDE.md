# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UCS321 EST Project — **Statement #4 (Biocon Ltd.)**: classify red blood cell images as Parasitized or Uninfected using a CNN, with Grad-CAM interpretability and a Streamlit dashboard.

Dataset: NIH Malaria Cell Images (27,558 images, 50% Parasitized / 50% Uninfected).

## Commands

```bash
# Pipeline (run in order)
PYTHONPATH=. python src/data_loader.py        # build & save train/val/test split manifests
PYTHONPATH=. python src/generate_flow_diagram.py
PYTHONPATH=. python src/train.py              # trains baseline_cnn + resnet50_transfer
PYTHONPATH=. python src/evaluate.py           # confusion matrix, ROC, PR, model comparison
PYTHONPATH=. python src/gradcam.py            # 4×4 Grad-CAM grids for TP/TN/FP/FN

# Dashboard
streamlit run dashboard/app.py

# Notebooks
jupyter notebook notebooks/01_EDA.ipynb
# notebooks/02_Colab_Training.ipynb is meant to run on Google Colab (T4 GPU)
```

No tests exist yet. `tests/` directory is empty placeholder.

## Architecture

```
data/raw/cell_images/{Parasitized,Uninfected}/*.png  (NIH, 27,558 images)
  → src/data_loader.py: stratified 70/15/15 split, tf.data pipeline, augmentation
    → data/processed/{train,val,test}_manifest.csv
  → src/train.py: trains baseline_cnn (3 conv blocks) + resnet50_transfer (frozen + fine-tune)
    → models/{baseline_cnn,resnet50_transfer}.h5 + training_history.json
  → src/evaluate.py: load each model, score test set, generate plots + CSVs
    → reports/figures/*.png, reports/{evaluation_metrics,model_comparison}.csv
    → models/model_info.json (best model pointer)
  → src/gradcam.py: Grad-CAM grids on best model
    → reports/figures/gradcam_{true,false}_{positive,negative}.png
  → dashboard/app.py: Streamlit loads model_info.json's best_model, serves predictions + Grad-CAM
```

## Data Schema

**Input**: 128×128×3 RGB float32 image, normalized [0, 1].

**Target** `label`:
- 0 = Uninfected (healthy red blood cell)
- 1 = Parasitized (malaria-infected)

**Split**: 70/15/15 train/val/test, stratified on label, seed=42.

## Key Config

`config/config.yaml` holds image size, augmentation parameters, training hyperparameters, evaluation threshold, and Grad-CAM grid size. All pipeline modules import via `src/config.py`.

## Convention notes

- Every script is runnable as `PYTHONPATH=. python src/<file>.py`
- All paths come from `src.config` — never hardcode
- `models/*.h5` and `data/raw/`, `data/processed/cell_images/` are gitignored (large binaries)
- Flow diagram (`reports/flow_diagram.png`) is a **required deliverable** per the EST Project Statements PDF page 2

## Cross-project rules (inherited from `~/CLAUDE.md`)

- Never add `Co-Authored-By: Claude` lines to commits
- Chunked execution for big multi-file tasks (one logical unit per response)
- Active recall > passive reading (applies to study materials, not this code project)
