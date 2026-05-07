"""Streamlit dashboard — Malaria RBC classifier with Grad-CAM explanation.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASSES, IMG_SIZE, MODELS_DIR  # noqa: E402

st.set_page_config(
    page_title="Malaria RBC Classifier — UCS321",
    page_icon="🩸",
    layout="wide",
)

# --- Styling ---
st.markdown(
    """
    <style>
    .header-card {
        background: linear-gradient(135deg, #312e81 0%, #1e293b 100%);
        padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 1rem;
        border: 1px solid #2dd4bf;
    }
    .header-card h1 { color: #fbbf24; margin: 0 0 0.4rem 0; }
    .header-card p { color: #e2e8f0; margin: 0; }
    .metric-card {
        background: #1e293b; padding: 1rem; border-radius: 0.5rem;
        border: 1px solid #334155;
    }
    .pred-card {
        padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: bold;
    }
    .pred-parasitized { background: #7f1d1d; color: #fecaca; }
    .pred-uninfected  { background: #064e3b; color: #a7f3d0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header-card">
      <h1>🩸 Malaria RBC Classifier</h1>
      <p>UCS321 EST Project · Statement #4 (Biocon Ltd.) · CNN + Grad-CAM interpretability</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Heavy imports + model loading (cached) ---
@st.cache_resource(show_spinner=True)
def load_model_and_info():
    import tensorflow as tf  # type: ignore

    info_path = MODELS_DIR / "model_info.json"
    if not info_path.exists():
        return None, None
    info = json.loads(info_path.read_text())
    model = tf.keras.models.load_model(info["weights_path"], compile=False)
    return model, info


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def gradcam_overlay_pil(model, arr: np.ndarray) -> Image.Image:
    from src.gradcam import gradcam_overlay
    overlay = gradcam_overlay(model, arr)
    return Image.fromarray(overlay)


# --- Sidebar: model info ---
with st.sidebar:
    st.markdown("## Model")
    model, info = load_model_and_info()
    if info is None:
        st.warning(
            "No trained model found. Train via `notebooks/02_Colab_Training.ipynb` "
            "and copy weights to `models/`."
        )
    else:
        st.markdown(f"**Best model:** `{info['best_model']}`")
        st.metric("Test AUC", f"{info['test_auc']:.4f}")
        st.metric("Test Accuracy", f"{info['test_accuracy']:.2%}")
        st.markdown("---")
        st.markdown("**All models**")
        for m in info["all_models"]:
            st.caption(
                f"{m['model']} — AUC {m['test_auc']:.3f}, F1 {m['test_f1']:.3f}"
            )

# --- Main: upload + predict ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Upload an RBC image")
    uploaded = st.file_uploader(
        "PNG / JPG of a single red blood cell",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)

with col_right:
    st.markdown("### Prediction")
    if uploaded is None:
        st.info("Upload an image to see prediction + Grad-CAM heatmap.")
    elif model is None:
        st.error("Cannot predict — no trained model loaded.")
    else:
        arr = preprocess_image(img)
        prob_parasitized = float(model.predict(arr[None, ...], verbose=0)[0][0])
        pred_idx = int(prob_parasitized >= 0.5)
        pred_label = CLASSES[pred_idx]
        confidence = prob_parasitized if pred_idx == 1 else (1 - prob_parasitized)

        cls = "pred-parasitized" if pred_idx == 1 else "pred-uninfected"
        emoji = "🦠" if pred_idx == 1 else "✅"
        st.markdown(
            f"""<div class="pred-card {cls}">
            <h2>{emoji} {pred_label}</h2>
            <p>Confidence: {confidence:.1%}</p></div>""",
            unsafe_allow_html=True,
        )
        st.progress(prob_parasitized, text=f"P(Parasitized) = {prob_parasitized:.4f}")

        st.markdown("### Grad-CAM heatmap")
        with st.spinner("Generating heatmap..."):
            overlay = gradcam_overlay_pil(model, arr)
        st.image(
            overlay,
            caption="Red regions = strongest evidence for the prediction",
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "Dataset: NIH Malaria Cell Images (27,558 images). "
    "Models: custom baseline CNN + ResNet50 transfer learning. "
    "Built for UCS321 — AI for Engineers."
)
