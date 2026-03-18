from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSample:
    frame: np.ndarray  # BGR uint8
    index: int
    timestamp_sec: float


def read_video_frames(
    video_path: str | Path,
    num_frames: int = 16,
    resize: int = 224,
    evenly_spaced: bool = True,
) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video has no frames.")

    if evenly_spaced:
        indices = np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=int)
    else:
        indices = np.arange(0, min(frame_count, num_frames), dtype=int)

    samples: list[FrameSample] = []
    for idx in indices.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if resize is not None:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
        t = float(idx / fps) if fps > 0 else 0.0
        samples.append(FrameSample(frame=frame, index=int(idx), timestamp_sec=t))

    cap.release()
    return samples
