from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aigenerated_detector.config import InferenceConfig
from aigenerated_detector.data.transforms import build_eval_transforms
from aigenerated_detector.models.image_classifier import ImageBinaryClassifier, load_checkpoint
from aigenerated_detector.utils.device import get_device


@dataclass(frozen=True)
class Prediction:
    p_fake: float
    label: str
    confidence: float


def score_to_label(p_fake: float, cfg: InferenceConfig) -> Prediction:
    if p_fake >= cfg.thresholds.likely_ai:
        return Prediction(p_fake=p_fake, label="Likely AI-Generated", confidence=p_fake)
    if p_fake >= cfg.thresholds.suspicious:
        # uncertainty region: confidence is closeness to 0.5 (lower is more uncertain)
        return Prediction(p_fake=p_fake, label="Suspicious", confidence=float(1.0 - abs(p_fake - 0.5) * 2.0))
    return Prediction(p_fake=p_fake, label="Likely Real", confidence=float(1.0 - p_fake))


@torch.inference_mode()
def predict_image(
    image_path: str | Path,
    checkpoint_path: str,
    model_name: str,
    cfg: InferenceConfig,
    device: torch.device | None = None,
) -> tuple[Prediction, torch.Tensor, np.ndarray]:
    device = device or get_device(prefer_gpu=False)

    model = ImageBinaryClassifier(model_name=model_name, pretrained=False).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image).astype(np.float32) / 255.0

    transform = build_eval_transforms(cfg.image_size)
    x = transform(image).unsqueeze(0).to(device)
    out = model(x)
    p_fake = float(out.probs.item())
    pred = score_to_label(p_fake, cfg)
    return pred, x, rgb
