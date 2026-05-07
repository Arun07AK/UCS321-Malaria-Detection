"""Grad-CAM interpretability.

Generates a 4x4 grid of test images per category (TP / TN / FP / FN) for the
best model with Grad-CAM heatmaps overlaid. Also exposes `gradcam_overlay()`
as a single-image utility used by the dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASSES, FIGURES_DIR, IMG_SIZE, MODELS_DIR, set_seeds
from src.data_loader import load_split_manifests


def _last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    raise ValueError("No Conv2D layer found in model")


def gradcam_heatmap(
    model: tf.keras.Model, img: np.ndarray, last_conv_name: str | None = None
) -> np.ndarray:
    """Returns a [H, W] heatmap in [0, 1] for a single image (already preprocessed)."""
    if last_conv_name is None:
        last_conv_name = _last_conv_layer_name(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )
    img_batch = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out, prediction = grad_model(img_batch)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., None]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def gradcam_overlay(
    model: tf.keras.Model, img: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Returns an RGB uint8 image with Grad-CAM heatmap overlaid."""
    heatmap = gradcam_heatmap(model, img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = np.uint8(img * 255) if img.dtype != np.uint8 else img
    overlay = np.uint8((1 - alpha) * base + alpha * colored)
    return overlay


def _load_test_image(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(im, dtype=np.float32) / 255.0


def _categorize(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1: return "true_positive"
    if y_true == 0 and y_pred == 0: return "true_negative"
    if y_true == 0 and y_pred == 1: return "false_positive"
    if y_true == 1 and y_pred == 0: return "false_negative"
    return "unknown"


def build_grid(
    model: tf.keras.Model, samples: list[tuple[str, int, int, float]], title: str, out_path: Path
) -> None:
    n = min(len(samples), 16)
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for ax in axes.flat: ax.axis("off")
    for ax, (path, y_true, y_pred, prob) in zip(axes.flat, samples[:n]):
        img = _load_test_image(path)
        overlay = gradcam_overlay(model, img)
        ax.imshow(overlay)
        actual = CLASSES[y_true]; predicted = CLASSES[y_pred]
        ax.set_title(f"actual: {actual}\npred: {predicted} ({prob:.2f})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seeds()
    info_path = MODELS_DIR / "model_info.json"
    if not info_path.exists():
        raise RuntimeError("Run src/evaluate.py first to pick the best model.")
    info = json.loads(info_path.read_text())
    weights = info["weights_path"]
    print(f"Loading best model: {info['best_model']} ({weights})")
    model = tf.keras.models.load_model(weights, compile=False)

    test_df = load_split_manifests()["test"]
    test_df = test_df.sample(min(len(test_df), 600), random_state=42).reset_index(drop=True)

    buckets = {"true_positive": [], "true_negative": [], "false_positive": [], "false_negative": []}
    print("Scoring test samples for Grad-CAM grids...")
    for _, row in test_df.iterrows():
        img = _load_test_image(row["path"])
        prob = float(model.predict(img[None, ...], verbose=0)[0][0])
        y_pred = int(prob >= 0.5)
        cat = _categorize(int(row["label"]), y_pred)
        if len(buckets[cat]) < 16:
            buckets[cat].append((row["path"], int(row["label"]), y_pred, prob))
        if all(len(v) >= 16 for v in buckets.values()):
            break

    for cat, samples in buckets.items():
        if not samples: continue
        out = FIGURES_DIR / f"gradcam_{cat}.png"
        title = f"Grad-CAM — {cat.replace('_', ' ').title()}  ({info['best_model']})"
        build_grid(model, samples, title, out)
        print(f"  Saved {out} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
