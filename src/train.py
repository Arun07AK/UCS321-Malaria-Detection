"""End-to-end training driver.

Trains both models in sequence and saves weights + history to models/.
Designed to run on Google Colab (T4 GPU) — see notebooks/02_Colab_Training.ipynb.

Run locally (slow on CPU; useful only for sanity checks):
    PYTHONPATH=. python src/train.py [--baseline-only]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from src.config import CFG, MODELS_DIR, REPORTS_DIR, set_seeds
from src.data_loader import get_datasets
from src.models import (
    build_baseline_cnn,
    build_resnet50_transfer,
    compile_for_training,
    unfreeze_last_block,
)


def _callbacks(weights_path: Path) -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=CFG["training"]["early_stopping_patience"],
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]


def train_baseline(ds: dict) -> dict:
    print("\n[1/2] Training baseline CNN")
    weights_path = MODELS_DIR / "baseline_cnn.h5"
    model = compile_for_training(build_baseline_cnn())
    t0 = time.time()
    history = model.fit(
        ds["train"],
        validation_data=ds["val"],
        epochs=CFG["training"]["epochs_baseline"],
        callbacks=_callbacks(weights_path),
        verbose=2,
    )
    elapsed = time.time() - t0
    return {
        "name": "baseline_cnn",
        "weights": str(weights_path),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "train_seconds": elapsed,
    }


def train_resnet50(ds: dict) -> dict:
    print("\n[2/2] Training ResNet50 transfer (frozen → fine-tune)")
    weights_path = MODELS_DIR / "resnet50_transfer.h5"
    model = compile_for_training(build_resnet50_transfer())

    t0 = time.time()
    h_frozen = model.fit(
        ds["train"],
        validation_data=ds["val"],
        epochs=CFG["training"]["epochs_transfer_frozen"],
        callbacks=_callbacks(weights_path),
        verbose=2,
    )

    print("\n  Stage 2: fine-tune last ResNet block")
    unfreeze_last_block(model)
    compile_for_training(model, learning_rate=CFG["training"]["fine_tune_lr"])
    h_finetune = model.fit(
        ds["train"],
        validation_data=ds["val"],
        epochs=CFG["training"]["epochs_transfer_finetune"],
        callbacks=_callbacks(weights_path),
        verbose=2,
    )
    elapsed = time.time() - t0

    combined = {}
    for k in h_frozen.history:
        combined[k] = [float(v) for v in h_frozen.history[k]] + [
            float(v) for v in h_finetune.history.get(k, [])
        ]
    return {
        "name": "resnet50_transfer",
        "weights": str(weights_path),
        "history": combined,
        "train_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--resnet-only", action="store_true")
    args = parser.parse_args()

    set_seeds()
    ds = get_datasets()

    results: list[dict] = []
    if not args.resnet_only:
        results.append(train_baseline(ds))
    if not args.baseline_only:
        results.append(train_resnet50(ds))

    out = MODELS_DIR / "training_history.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved training history → {out}")


if __name__ == "__main__":
    main()
