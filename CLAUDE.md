# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

UCS321 EST Project — **Statement #4 (Biocon Ltd.)**: classify red blood cell images as Parasitized or Uninfected using a CNN, with Grad-CAM interpretability and a Streamlit dashboard.

**Status: shipped.** Code, training, evaluation, Grad-CAM, presentation deck, and mobile cheatsheet all complete and pushed to `Arun07AK/UCS321-Malaria-Detection`. Final test-set result: **95.50% accuracy** (Baseline CNN beat ResNet50 transfer at 94.53%).

Dataset: NIH Malaria Cell Images (27,558 images, 50% Parasitized / 50% Uninfected).

## Live URLs

- **GitHub:** https://github.com/Arun07AK/UCS321-Malaria-Detection
- **Mobile cheatsheet (viva walkthrough):** https://ucs321-malaria-cheatsheet.vercel.app
- **Plain-language explainer:** https://ucs321-malaria-cheatsheet.vercel.app/explainer

Cheatsheet is auto-deployed via Vercel CLI from `cheatsheet/` (linked to project `arun07aks-projects/ucs321-malaria-cheatsheet`).

## Local state caveats

- **`models/` is empty.** Training was done on Colab T4. Only the `reports/` artifacts (figures + CSVs) were downloaded. To run `dashboard/app.py` locally, the user must either:
  1. Re-pull the `.keras` weights from Colab via `notebooks/02_Colab_Training.ipynb`, or
  2. Re-train locally with `python src/train.py` (slow on CPU, fast on Apple MPS).

  The `.keras` files are intentionally gitignored — ResNet50 alone is ~95 MB, exceeding GitHub's 100 MB limit.
- **Dataset:** `data/raw/cell_images/` is present locally (27,558 PNGs, ~340 MB) but gitignored.
- **Final results** in `reports/model_comparison.csv` are real (from Colab run on 2026-05-08).

## Commands

```bash
# Pipeline (run in order — assumes data/raw/ and Kaggle creds)
PYTHONPATH=. python src/data_loader.py        # build & save train/val/test split manifests
PYTHONPATH=. python src/generate_flow_diagram.py  # render reports/flow_diagram.png
PYTHONPATH=. python src/train.py              # trains baseline_cnn + resnet50_transfer (~45 min Colab T4)
PYTHONPATH=. python src/evaluate.py           # confusion matrix, ROC, PR, model comparison
PYTHONPATH=. python src/gradcam.py            # 4×4 Grad-CAM grids for TP/TN/FP/FN

# Dashboard (requires models/*.keras + models/model_info.json — see "Local state caveats")
streamlit run dashboard/app.py

# Notebooks
jupyter notebook notebooks/01_EDA.ipynb
# notebooks/02_Colab_Training.ipynb is meant to run on Google Colab (T4 GPU)

# Cheatsheet — preview locally / re-deploy
open cheatsheet/index.html
cd cheatsheet && vercel deploy --prod --yes   # already linked; re-deploys to ucs321-malaria-cheatsheet.vercel.app
```

No tests exist. `tests/` is an empty placeholder.

## Architecture

```
data/raw/cell_images/{Parasitized,Uninfected}/*.png  (NIH, 27,558 images)
  → src/data_loader.py: stratified 70/15/15 split, tf.data pipeline, augmentation
    → data/processed/{train,val,test}_manifest.csv
  → src/train.py: trains baseline_cnn (3 conv blocks) + resnet50_transfer (frozen + fine-tune)
    → models/{baseline_cnn,resnet50_transfer}.keras + training_history.json
  → src/evaluate.py: load each model, score test set, generate plots + CSVs
    → reports/figures/*.png, reports/{evaluation_metrics,model_comparison}.csv
    → models/model_info.json (best model pointer)
  → src/gradcam.py: Grad-CAM grids on best model
    → reports/figures/gradcam_{true,false}_{positive,negative}.png
  → dashboard/app.py: Streamlit loads model_info.json's best_model, serves predictions + Grad-CAM
  → presentation/presentation.html: 12-slide TICSR-style interactive deck (4 canvas demos)
  → cheatsheet/{index.html,explainer.html}: mobile viva reference + plain-language explainer
```

## Important: model preprocessing fix

`src/models.py` originally used a `Lambda` layer to wrap ResNet50's `preprocess_input`. Keras 3 (TF 2.16+) refuses to deserialize Python lambdas without `safe_mode=False`, so saving and loading the trained ResNet50 broke. **Current code uses a plain `Rescaling(255.0)` layer instead** — fully serializable, the fine-tune stage absorbs the missing BGR/mean-subtraction offset. Don't reintroduce Lambda layers.

## Data Schema

**Input**: 128×128×3 RGB float32 image, normalized [0, 1].

**Target** `label`:
- 0 = Uninfected (healthy red blood cell)
- 1 = Parasitized (malaria-infected)

**Split**: 70/15/15 train/val/test, stratified on label, seed=42 → train 19,290 / val 4,133 / test 4,135.

## Key Config

`config/config.yaml` holds image size, augmentation parameters, training hyperparameters, evaluation threshold, and Grad-CAM grid size. All pipeline modules import via `src/config.py`.

## Convention notes

- Every script is runnable as `PYTHONPATH=. python src/<file>.py`
- All paths come from `src.config` — never hardcode
- `models/*.keras`, `models/*.h5`, `models/*.json`, `data/raw/`, `data/processed/*` (except `.gitkeep`) are all gitignored — large or machine-specific
- Flow diagram (`reports/flow_diagram.png`) is a **required deliverable** per the EST Project Statements PDF page 2
- Cheatsheet visual style is lifted from `~/AEKAY/RESEARCH/A Study and Development of One-Class Classification Techniques for Text Data/cheatsheet_vercel/` — match its palette/typography if making sibling pages
- Presentation visual style is lifted from `~/AEKAY/RESEARCH/A Study and Development of One-Class Classification Techniques for Text Data/TICSR_2026_Interactive_Deck.html` — same palette + interactive canvas demo pattern

## Repo layout

```
ai_project/
├── data/                       # raw + processed (gitignored except .gitkeep)
├── notebooks/                  # 01_EDA, 02_Colab_Training
├── src/                        # config, data_loader, models, train, evaluate, gradcam, generate_flow_diagram
├── models/                     # weights (gitignored — too big for GitHub)
├── reports/figures/            # all evaluation plots (committed)
├── dashboard/app.py            # Streamlit
├── presentation/presentation.html  # 12-slide interactive deck
├── cheatsheet/                 # Vercel-deployed mobile reference
│   ├── index.html              # viva cheatsheet (jargon, accordions)
│   ├── explainer.html          # plain-language story (no jargon)
│   └── vercel.json             # cleanUrls + headers
├── config/config.yaml
├── docs/flow_diagram.md
└── EST Project Statements_*.pdf  # original brief
```

## Cross-project rules (inherited from `~/CLAUDE.md`)

- Never add `Co-Authored-By: Claude` lines to commits
- Chunked execution for big multi-file tasks (one logical unit per response)
- PYQ integrity rule (doesn't apply here — no PYQs in this project)
