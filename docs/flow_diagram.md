# Project Flow Diagram

UCS321 EST Project — Statement #4 (Biocon Malaria RBC Classification)

```mermaid
flowchart TD
    A["NIH Malaria Cell Images<br/>27,558 RBC images<br/>50% Parasitized · 50% Uninfected"] --> B
    B["Pre-processing<br/>• Resize 128x128<br/>• Normalize 0-1<br/>• Stratified 70/15/15 split"] --> C
    C["Data Augmentation<br/>• H/V flip · rotation<br/>• Brightness · contrast<br/>• Gaussian noise"] --> D1 & D2
    D1["Baseline CNN<br/>3 conv blocks + dense"] --> E
    D2["ResNet50 transfer<br/>frozen + fine-tune"] --> E
    E["Training<br/>Adam · binary crossentropy<br/>EarlyStopping · ModelCheckpoint"] --> F
    F["Evaluation<br/>• Accuracy · Precision · Recall · F1<br/>• ROC-AUC · PR curve<br/>• Confusion matrix"] --> G
    G["Grad-CAM Interpretability<br/>4x4 grids: TP · TN · FP · FN"] --> H
    H["Streamlit Dashboard<br/>Upload image -> prediction + heatmap"]

    classDef pre fill:#1e293b,stroke:#2dd4bf,color:#fff
    classDef model fill:#312e81,stroke:#fbbf24,color:#fff
    classDef out fill:#064e3b,stroke:#34d399,color:#fff
    class A,B,C pre
    class D1,D2,E model
    class F,G,H out
```

## Pre-processing steps

1. **Load** images from `data/raw/cell_images/{Parasitized,Uninfected}/`
2. **Resize** to 128×128 px (square, RGB)
3. **Normalize** pixel values to [0, 1] by dividing by 255
4. **Split** into train/val/test (70/15/15) with stratification on label
5. **Augment** training images: random flips, ±20° rotation, zoom, brightness/contrast jitter, low-amplitude Gaussian noise (addresses "noise" + "variability" in problem statement)

## Visualization steps

- **EDA notebook** — class balance bar plot, image-size histogram, sample image grid (8 of each class), augmented-vs-original comparison
- **Training curves** — loss/accuracy/AUC per epoch for both models
- **Confusion matrix** + **ROC curve** + **Precision-Recall curve** per model
- **Grad-CAM heatmaps** — overlaid on test images to highlight regions the model focuses on
- **Streamlit dashboard** — live prediction + Grad-CAM overlay for any uploaded image
