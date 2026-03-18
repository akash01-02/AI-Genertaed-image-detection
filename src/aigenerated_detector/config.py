from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    likely_ai: float = 0.70
    suspicious: float = 0.40


@dataclass(frozen=True)
class InferenceConfig:
    image_size: int = 224
    video_num_frames: int = 16
    video_frame_stride: int = 1
    thresholds: Thresholds = Thresholds()


DEFAULT_MODEL_NAME = "tf_efficientnet_b0"
