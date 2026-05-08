# UCS321-Malaria-Detection

> **UCS321 — AI for Engineers · EST Project (Statement #4: Biocon Ltd.)**
> *Develop an advanced CNN model to classify red blood cell images as healthy or malaria-infected, addressing class imbalance, image noise, and variability, while incorporating model interpretability (e.g., Grad-CAM) to assist medical experts in validating predictions.*

A complete deep-learning pipeline that classifies red blood cell images as **Parasitized** or **Uninfected**, using a custom CNN baseline and a ResNet50 transfer-learning model. Includes Grad-CAM interpretability and a Streamlit dashboard.

---

## Quick References

- **Phone cheatsheet (live):** https://ucs321-malaria-cheatsheet.vercel.app — viva walkthrough, mobile-optimized
- **Presentation deck:** [`presentation/presentation.html`](./presentation/presentation.html) — interactive 12-slide deck
- **Flow diagram:** [`reports/flow_diagram.png`](./reports/flow_diagram.png) — required deliverable per project brief
- **Results table:** [`reports/model_comparison.csv`](./reports/model_comparison.csv) — final test-set metrics

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (NIH Malaria Cell Images, ~340 MB)
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p data/raw --unzip

# 3. Build splits + run EDA
PYTHONPATH=. python src/data_loader.py
jupyter notebook notebooks/01_EDA.ipynb

# 4. Train (recommended: Google Colab T4 GPU — see notebooks/02_Colab_Training.ipynb)
PYTHONPATH=. python src/train.py

# 5. Evaluate + Grad-CAM
PYTHONPATH=. python src/evaluate.py
PYTHONPATH=. python src/gradcam.py

# 6. Run the dashboard
streamlit run dashboard/app.py
```

---

## Repository Layout

```
ai_project/
├── data/
│   ├── raw/cell_images/        # NIH dataset (gitignored)
│   └── processed/              # train/val/test split manifests
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Colab_Training.ipynb # full pipeline runnable on free T4 GPU
├── src/
│   ├── config.py               # YAML loader, seeding, paths
│   ├── data_loader.py          # tf.data pipeline + 70/15/15 stratified split
│   ├── models.py               # baseline CNN + ResNet50 transfer learning
│   ├── train.py                # training driver with EarlyStopping
│   ├── evaluate.py             # confusion matrix · ROC · PR · classification report
│   ├── gradcam.py              # Grad-CAM heatmap utility + 4×4 grids
│   └── generate_flow_diagram.py
├── models/                     # weights + model_info.json (gitignored .h5)
├── reports/
│   ├── flow_diagram.png        # mandatory deliverable per project brief
│   ├── figures/                # all evaluation plots
│   ├── evaluation_metrics.csv
│   └── model_comparison.csv
├── dashboard/app.py            # Streamlit upload → predict → Grad-CAM
├── presentation/presentation.html
├── config/config.yaml          # all hyperparameters in one place
└── docs/flow_diagram.md
```

---

## Pipeline

1. **Data** — NIH Malaria Cell Images (27,558 images, 50/50 balanced)
2. **Pre-processing** — resize 128×128 RGB, normalize [0, 1], stratified 70/15/15 split
3. **Augmentation** — H/V flip, rotation, brightness, contrast, Gaussian noise
   *(directly addresses "image noise" and "variability" in the problem statement)*
4. **Models**
   - **Baseline CNN**: 3 conv blocks + global average pooling + dense head (~100K params)
   - **ResNet50 Transfer**: ImageNet pretrained, frozen base → fine-tune last block (~24M params)
5. **Training** — Adam, binary crossentropy, EarlyStopping on `val_auc`, ReduceLROnPlateau
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
7. **Interpretability** — Grad-CAM heatmaps for true/false × pos/neg samples (per problem brief)
8. **Dashboard** — Streamlit app with image upload + prediction + Grad-CAM overlay

See `reports/flow_diagram.png` for the visual pipeline.

---

## How the project addresses the problem brief

| Brief requirement | Implementation |
|---|---|
| "Advanced CNN model" | ResNet50 transfer learning + custom baseline for comparison |
| "Class imbalance" | Stratified split + class-weighted loss capability + per-class metrics |
| "Image noise" | Augmentation includes Gaussian noise + brightness/contrast jitter |
| "Variability" | Multi-augmentation pipeline (flip, rotation, zoom) |
| "Model interpretability (Grad-CAM)" | Dedicated `src/gradcam.py` + dashboard overlay + 4×4 grids in report |
| Mandatory flow diagram | `reports/flow_diagram.png` (rendered from `docs/flow_diagram.md`) |

---

## Results

Trained on Google Colab (T4 GPU). Test set: 4,135 images held out from training.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Baseline CNN** ⭐ | **95.50%** | 95.52% | 95.50% | **0.9550** | **0.9883** | **0.9888** |
| ResNet50 Transfer | 94.53% | 94.64% | 94.53% | 0.9453 | 0.9871 | 0.9876 |

**Best model: Baseline CNN** — surprisingly, the small custom architecture (~100K params) edged out ResNet50 transfer (~24M params) by ~1 point on every metric. Likely reason: ResNet50's ImageNet feature distribution is far from microscopy stains, so its inductive bias works against it; the lightweight CNN learns task-specific features without the baggage.

Confusion matrix (Baseline CNN): 1,995 true negatives · 1,954 true positives · 73 false positives · 113 false negatives. Per-class F1: Uninfected 0.9555, Parasitized 0.9546.

All figures (training curves, ROC, PR, confusion matrices, Grad-CAM grids) are in `reports/figures/`.

## Tech Stack

- **TensorFlow / Keras 2.15+** — model definitions & training
- **scikit-learn** — splits, metrics, classification report
- **OpenCV / Pillow** — image I/O, Grad-CAM colormap overlay
- **Streamlit** — dashboard
- **matplotlib / seaborn** — all plots
- **Kaggle CLI** — dataset download

---

## License

MIT (educational use).

## Author

**Arun AK** — UCS321 (AI for Engineers), B.E. CSE, Thapar University.
