from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from aigenerated_detector.config import DEFAULT_MODEL_NAME, InferenceConfig, Thresholds
from aigenerated_detector.explainability.gradcam import compute_gradcam
from aigenerated_detector.inference.predict_image import predict_image
from aigenerated_detector.inference.predict_video import predict_video

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch




st.set_page_config(page_title="Deepfake Detector + Grad-CAM", layout="wide")
st.title("AI-Generated Image & Deepfake Video Detection")
st.caption("Upload an image or video to get a prediction + Grad-CAM explanations.")


def _thresholds_ui() -> Thresholds:
    st.sidebar.subheader("Thresholds")
    likely_ai = st.sidebar.slider("Likely AI-Generated threshold", 0.5, 0.95, 0.70, 0.01)
    suspicious = st.sidebar.slider("Suspicious threshold", 0.1, 0.8, 0.40, 0.01)
    if suspicious > likely_ai:
        st.sidebar.warning("Suspicious threshold should be <= Likely AI threshold")
    return Thresholds(likely_ai=float(likely_ai), suspicious=float(suspicious))


st.sidebar.header("Model")
checkpoint_path = st.sidebar.text_input("Checkpoint path", value="outputs/image_run/best.pt")
model_name = st.sidebar.text_input("Model name (timm)", value=DEFAULT_MODEL_NAME)
image_size = st.sidebar.selectbox("Image size", options=[224, 256, 299], index=0)
thresholds = _thresholds_ui()

cfg = InferenceConfig(image_size=int(image_size), thresholds=thresholds)

st.sidebar.header("Video")
num_frames = st.sidebar.slider("Sampled frames", 4, 32, 16, 1)
cfg = InferenceConfig(image_size=int(image_size), video_num_frames=int(num_frames), thresholds=thresholds)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Upload")
    upload = st.file_uploader("Upload an image (.jpg/.png) or video (.mp4)", type=["jpg", "jpeg", "png", "mp4"])

with col_right:
    st.subheader("Result")
    if upload is None:
        st.info("Upload a file to begin.")
        st.stop()

    suffix = Path(upload.name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".mp4"}:
        st.error("Unsupported file type")
        st.stop()

    if not Path(checkpoint_path).exists():
        st.warning("Checkpoint not found yet. Train a model or point to an existing .pt file.")

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / upload.name
        tmp_path.write_bytes(upload.getbuffer())

        device = torch.device("cpu")

        if suffix in {".jpg", ".jpeg", ".png"}:
            pred, x, rgb = predict_image(tmp_path, checkpoint_path, model_name, cfg, device=device)
            st.metric("Label", pred.label)
            st.metric("Confidence", f"{pred.confidence:.3f}")
            st.write({"p_fake": pred.p_fake})

            st.image(str(tmp_path), caption="Input", use_container_width=True)

            if st.checkbox("Show Grad-CAM", value=True):
                # compute_gradcam requires input_rgb_float in [0,1]
                from aigenerated_detector.models.image_classifier import ImageBinaryClassifier, load_checkpoint

                model = ImageBinaryClassifier(model_name=model_name, pretrained=False).to(device)
                load_checkpoint(model, checkpoint_path, device)
                cam = compute_gradcam(model, x, rgb)
                st.image(cam.overlay_rgb, caption="Grad-CAM overlay", use_container_width=True)

        else:
            video_pred, frame_preds = predict_video(tmp_path, checkpoint_path, model_name, cfg, device=device)
            st.metric("Video label", video_pred.label)
            st.metric("Video confidence", f"{video_pred.confidence:.3f}")
            st.write({"video_p_fake": video_pred.p_fake, "num_frames": len(frame_preds)})

            # Sort frames by p_fake, show top suspicious frames
            sorted_frames = sorted(frame_preds, key=lambda f: f.p_fake, reverse=True)
            top_k = st.slider("Show top-K frames", 1, min(12, len(sorted_frames)), min(6, len(sorted_frames)))

            show_cam = st.checkbox("Show Grad-CAM on frames", value=True)
            from aigenerated_detector.models.image_classifier import ImageBinaryClassifier, load_checkpoint

            model = ImageBinaryClassifier(model_name=model_name, pretrained=False).to(device)
            load_checkpoint(model, checkpoint_path, device)

            grid_cols = 3
            rows = (top_k + grid_cols - 1) // grid_cols
            idx = 0
            for _ in range(rows):
                cols = st.columns(grid_cols)
                for c in cols:
                    if idx >= top_k:
                        break
                    fp = sorted_frames[idx]
                    title = f"frame={fp.frame_index}  t={fp.timestamp_sec:.2f}s  p_fake={fp.p_fake:.3f}"
                    if show_cam:
                        cam = compute_gradcam(model, fp.input_tensor, fp.rgb_float)
                        c.image(cam.overlay_rgb, caption=title, use_container_width=True)
                    else:
                        rgb_u8 = (fp.rgb_float * 255).astype(np.uint8)
                        c.image(rgb_u8, caption=title, use_container_width=True)
                    idx += 1
