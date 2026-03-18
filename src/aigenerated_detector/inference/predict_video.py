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
from aigenerated_detector.utils.video import FrameSample, read_video_frames
from aigenerated_detector.inference.predict_image import Prediction, score_to_label


@dataclass(frozen=True)
class VideoFramePrediction:
    frame_index: int
    timestamp_sec: float
    p_fake: float
    rgb_float: np.ndarray  # HxWx3 float in [0,1]
    input_tensor: torch.Tensor  # [1,3,H,W]


@torch.inference_mode()
def predict_video(
    video_path: str | Path,
    checkpoint_path: str,
    model_name: str,
    cfg: InferenceConfig,
    device: torch.device | None = None,
) -> tuple[Prediction, list[VideoFramePrediction]]:
    device = device or get_device(prefer_gpu=False)

    model = ImageBinaryClassifier(model_name=model_name, pretrained=False).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    frames: list[FrameSample] = read_video_frames(
        video_path,
        num_frames=cfg.video_num_frames,
        resize=cfg.image_size,
        evenly_spaced=True,
    )

    transform = build_eval_transforms(cfg.image_size)
    frame_preds: list[VideoFramePrediction] = []

    for s in frames:
        # cv2 gives BGR; convert to RGB
        rgb = s.frame[:, :, ::-1].copy()
        rgb_float = rgb.astype(np.float32) / 255.0
        pil = Image.fromarray(rgb)
        x = transform(pil).unsqueeze(0).to(device)
        out = model(x)
        p_fake = float(out.probs.item())
        frame_preds.append(
            VideoFramePrediction(
                frame_index=s.index,
                timestamp_sec=s.timestamp_sec,
                p_fake=p_fake,
                rgb_float=rgb_float,
                input_tensor=x,
            )
        )

    if not frame_preds:
        raise RuntimeError("No frames were read from the video.")

    video_p_fake = float(np.mean([fp.p_fake for fp in frame_preds]))
    video_pred = score_to_label(video_p_fake, cfg)
    return video_pred, frame_preds
