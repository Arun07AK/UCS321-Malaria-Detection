"""Central config loader. All pipeline modules import from here."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


CFG = load_config()

SEED = CFG["project"]["seed"]
IMG_SIZE = CFG["data"]["img_size"]
CHANNELS = CFG["data"]["channels"]
CLASSES = CFG["data"]["classes"]
NUM_CLASSES = len(CLASSES)

DATA_RAW = PROJECT_ROOT / CFG["paths"]["data_raw"]
DATA_PROCESSED = PROJECT_ROOT / CFG["paths"]["data_processed"]
MODELS_DIR = PROJECT_ROOT / CFG["paths"]["models"]
REPORTS_DIR = PROJECT_ROOT / CFG["paths"]["reports"]
FIGURES_DIR = PROJECT_ROOT / CFG["paths"]["figures"]

for d in (DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Make runs reproducible across numpy, python, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Image size:   {IMG_SIZE}x{IMG_SIZE}x{CHANNELS}")
    print(f"Classes:      {CLASSES}")
    print(f"Seed:         {SEED}")
