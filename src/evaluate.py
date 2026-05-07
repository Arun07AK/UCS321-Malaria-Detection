"""Evaluation: confusion matrix, ROC, PR curve, classification report.

Loads weights from models/, generates plots in reports/figures/, and writes
two CSVs:
  * reports/evaluation_metrics.csv  — per-model row with all key metrics
  * reports/model_comparison.csv    — same data sorted by test_auc

Picks the best model by test AUC and writes models/model_info.json — the
dashboard reads this to pick which model to load.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import CFG, CLASSES, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, set_seeds
from src.data_loader import get_datasets

MODEL_NAMES = ["baseline_cnn", "resnet50_transfer"]


def collect_predictions(model: tf.keras.Model, ds) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0).flatten()
        y_pred.extend(probs.tolist())
        y_true.extend(batch_y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_training_curves(history: dict, model_name: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{model_name} — training curves", fontsize=13, fontweight="bold")

    epochs = range(1, len(history.get("loss", [])) + 1)
    axes[0].plot(epochs, history.get("loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("epoch")

    axes[1].plot(epochs, history.get("accuracy", []), label="train")
    axes[1].plot(epochs, history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, history.get("auc", []), label="train")
    axes[2].plot(epochs, history.get("val_auc", []), label="val")
    axes[2].set_title("AUC"); axes[2].legend(); axes[2].set_xlabel("epoch")

    plt.tight_layout()
    out = FIGURES_DIR / f"training_curves_{model_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred_binary, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred_binary)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar=False,
    )
    ax.set_title(f"Confusion matrix — {model_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_pred, model_name: str) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}", color="#2dd4bf")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {model_name}"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"roc_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return float(roc_auc)


def plot_pr(y_true, y_pred, model_name: str) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, label=f"AUC = {pr_auc:.3f}", color="#fbbf24")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall — {model_name}"); ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"pr_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return float(pr_auc)


def evaluate_one(model_name: str, ds_test) -> dict:
    weights = MODELS_DIR / f"{model_name}.keras"
    if not weights.exists():
        print(f"  [skip] {weights} not found")
        return None

    model = tf.keras.models.load_model(weights, compile=False)
    y_true, y_pred = collect_predictions(model, ds_test)
    threshold = CFG["evaluation"]["threshold"]
    y_bin = (y_pred >= threshold).astype(int)

    report = classification_report(
        y_true, y_bin, target_names=CLASSES, output_dict=True, zero_division=0
    )

    plot_confusion(y_true, y_bin, model_name)
    roc_auc = plot_roc(y_true, y_pred, model_name)
    pr_auc = plot_pr(y_true, y_pred, model_name)

    return {
        "model": model_name,
        "test_accuracy": report["accuracy"],
        "test_precision": report["weighted avg"]["precision"],
        "test_recall": report["weighted avg"]["recall"],
        "test_f1": report["weighted avg"]["f1-score"],
        "test_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "uninfected_f1": report[CLASSES[0]]["f1-score"],
        "parasitized_f1": report[CLASSES[1]]["f1-score"],
    }


def main() -> None:
    set_seeds()
    ds = get_datasets()
    test = ds["test"]

    history_path = MODELS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            histories = json.load(f)
        for entry in histories:
            plot_training_curves(entry["history"], entry["name"])

    rows = []
    for name in MODEL_NAMES:
        print(f"Evaluating {name}...")
        row = evaluate_one(name, test)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No models found to evaluate. Train them first.")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "evaluation_metrics.csv", index=False)
    df_sorted = df.sort_values("test_auc", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nFinal comparison:\n", df_sorted.to_string(index=False))

    best = df_sorted.iloc[0].to_dict()
    info = {
        "best_model": best["model"],
        "weights_path": str(MODELS_DIR / f"{best['model']}.keras"),
        "test_auc": best["test_auc"],
        "test_accuracy": best["test_accuracy"],
        "all_models": df_sorted.to_dict(orient="records"),
    }
    with open(MODELS_DIR / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nBest model: {best['model']}  (AUC={best['test_auc']:.4f})")


if __name__ == "__main__":
    main()
