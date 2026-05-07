"""
Data loader for the NIH Malaria Cell Images dataset.

Expected raw layout (after `kaggle datasets download -d iarunava/cell-images-for-detecting-malaria`):
    data/raw/cell_images/
        Parasitized/  *.png
        Uninfected/   *.png

This module:
  1. Walks the raw directory and produces a stratified 70/15/15 split.
  2. Saves split manifests as CSVs to data/processed/.
  3. Builds tf.data.Dataset pipelines (load → resize → augment → batch → prefetch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    CFG,
    CLASSES,
    DATA_PROCESSED,
    DATA_RAW,
    IMG_SIZE,
    SEED,
    set_seeds,
)

AUTOTUNE = tf.data.AUTOTUNE


def build_manifest(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Walk Parasitized/ and Uninfected/ folders, return DataFrame[path, label]."""
    rows = []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected {class_dir} to exist. Run dataset download first."
            )
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                rows.append({"path": str(img_path), "label": label_idx})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def split_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split."""
    train_frac = CFG["data"]["train_split"]
    val_frac = CFG["data"]["val_split"]
    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df["label"], random_state=SEED
    )
    rel_val = val_frac / (1.0 - train_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val, stratify=temp_df["label"], random_state=SEED
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split_manifests() -> dict[str, pd.DataFrame]:
    df = build_manifest()
    train_df, val_df, test_df = split_manifest(df)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED / "train_manifest.csv", index=False)
    val_df.to_csv(DATA_PROCESSED / "val_manifest.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test_manifest.csv", index=False)
    return {"train": train_df, "val": val_df, "test": test_df}


def load_split_manifests() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "val", "test"):
        p = DATA_PROCESSED / f"{name}_manifest.csv"
        if not p.exists():
            return save_split_manifests()
        out[name] = pd.read_csv(p)
    return out


def _decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _augment(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    aug = CFG["augmentation"]
    if aug["horizontal_flip"]:
        img = tf.image.random_flip_left_right(img)
    if aug["vertical_flip"]:
        img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(
        img, max_delta=(aug["brightness_range"][1] - 1.0)
    )
    img = tf.image.random_contrast(
        img, lower=aug["brightness_range"][0], upper=aug["brightness_range"][1]
    )
    noise = tf.random.normal(
        tf.shape(img), mean=0.0, stddev=aug["gaussian_noise_stddev"]
    )
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img, label


def make_dataset(
    df: pd.DataFrame, batch_size: int, shuffle: bool, augment: bool
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 4096), seed=SEED)
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def get_datasets(batch_size: int | None = None) -> dict[str, tf.data.Dataset]:
    """Return train/val/test tf.data pipelines built from saved manifests."""
    if batch_size is None:
        batch_size = CFG["training"]["batch_size"]
    splits = load_split_manifests()
    return {
        "train": make_dataset(splits["train"], batch_size, shuffle=True, augment=True),
        "val": make_dataset(splits["val"], batch_size, shuffle=False, augment=False),
        "test": make_dataset(splits["test"], batch_size, shuffle=False, augment=False),
        "manifests": splits,
    }


if __name__ == "__main__":
    set_seeds()
    splits = load_split_manifests()
    for name, df in splits.items():
        counts = df["label"].value_counts().to_dict()
        labelled = {CLASSES[k]: v for k, v in counts.items()}
        print(f"{name:5s}  total={len(df):>6}  by class={labelled}")
