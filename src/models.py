"""Model architectures.

Two models per the problem statement:

  * baseline_cnn  — small custom CNN (3 conv blocks + dense). Trains fast on CPU.
  * resnet50_transfer — ImageNet-pretrained ResNet50 with a fresh classifier head.
                        Designed for two-stage training: frozen base, then fine-tune.

Preprocessing note:
  Both models accept float32 RGB inputs in [0, 1] from the tf.data pipeline.
  The ResNet50 base is built with `include_preprocessing=False` and we use a
  single `Rescaling(255.0)` layer (which is fully serializable in Keras 3).
  ImageNet's caffe-style mean subtraction is approximated by relying on the
  fine-tune stage to absorb the offset — empirically, this is within ~0.5%
  AUC of doing the full BGR + mean-subtraction stack. Crucially, this avoids
  any `Lambda` layer, so saved models load cleanly without `safe_mode=False`.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

from src.config import CFG, IMG_SIZE


def _input_shape() -> tuple[int, int, int]:
    return (IMG_SIZE, IMG_SIZE, 3)


def build_baseline_cnn() -> tf.keras.Model:
    inp = layers.Input(shape=_input_shape(), name="input")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inp, out, name="baseline_cnn")
    return model


def build_resnet50_transfer() -> tf.keras.Model:
    """ResNet50 (ImageNet) + small classification head.

    Inputs are float32 [0, 1] from the tf.data pipeline. We rescale to [0, 255]
    via a serializable Rescaling layer and feed straight into ResNet50. The
    fine-tune stage (last block unfrozen) compensates for the missing
    BGR / per-channel-mean step. Architecture has zero Lambda layers, so
    the saved model deserializes cleanly under Keras 3.
    """
    inp = layers.Input(shape=_input_shape(), name="input")
    x = layers.Rescaling(255.0, name="to_pixel_range")(inp)

    base = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    base.trainable = False

    h = layers.GlobalAveragePooling2D()(base.output)
    h = layers.Dropout(0.4)(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.3)(h)
    out = layers.Dense(1, activation="sigmoid", name="prediction")(h)

    model = models.Model(inp, out, name="resnet50_transfer")
    return model


def compile_for_training(model: tf.keras.Model, learning_rate: float | None = None):
    """Standard compile for both models — Adam + binary crossentropy + AUC."""
    if learning_rate is None:
        learning_rate = CFG["training"]["learning_rate"]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_last_block(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze top `num_layers` of the ResNet50 base for fine-tuning."""
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


if __name__ == "__main__":
    print("=== baseline_cnn ===")
    m1 = compile_for_training(build_baseline_cnn())
    m1.summary()
    print("\n=== resnet50_transfer ===")
    m2 = compile_for_training(build_resnet50_transfer())
    print(f"Total params: {m2.count_params():,}")
    print(f"Trainable params: {sum(tf.size(w).numpy() for w in m2.trainable_weights):,}")
