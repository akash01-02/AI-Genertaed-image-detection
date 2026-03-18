from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as e:  # pragma: no cover
    GradCAM = None  # type: ignore
    show_cam_on_image = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class CamResult:
    heatmap: np.ndarray  # HxW float in [0,1]
    overlay_rgb: np.ndarray  # HxWx3 uint8


class _LogitsOnlyModel(torch.nn.Module):
    """Adapter for GradCAM: ensures model(input) returns a Tensor of logits."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "logits"):
            logits = getattr(out, "logits")
            if logits.ndim == 1:
                return logits.unsqueeze(1)
            return logits
        raise TypeError("Unsupported model output type for Grad-CAM")


def _require_gradcam() -> None:
    if GradCAM is None or show_cam_on_image is None:
        raise RuntimeError(
            "pytorch-grad-cam is required for Grad-CAM. "
            "Install with: pip install pytorch-grad-cam"
        ) from _IMPORT_ERROR


def find_target_layer(backbone: torch.nn.Module) -> torch.nn.Module:
    """Best-effort target layer selection for timm models."""
    # EfficientNet family
    for attr in ["conv_head", "bn2", "act2"]:
        if hasattr(backbone, attr):
            return getattr(backbone, attr)

    # ResNet family (timm/torchvision style)
    if hasattr(backbone, "layer4"):
        layer4 = getattr(backbone, "layer4")
        if hasattr(layer4, "__len__") and len(layer4) > 0:
            block = layer4[-1]
            for conv_attr in ["conv3", "conv2"]:
                if hasattr(block, conv_attr):
                    return getattr(block, conv_attr)
            return block

    # Fallback: last module
    modules = [m for m in backbone.modules()]
    return modules[-1]


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    input_rgb_float: np.ndarray,
    target_layers: list[torch.nn.Module] | None = None,
) -> CamResult:
    """Compute Grad-CAM for binary classifier.

    - input_tensor: shape [1,3,H,W]
    - input_rgb_float: HxWx3 float in [0,1]
    """
    _require_gradcam()

    model.eval()
    cam_model = _LogitsOnlyModel(model)
    cam_model.eval()

    if target_layers is None:
        # Most of our models are wrappers with .backbone
        backbone = model.backbone if hasattr(model, "backbone") else model
        target_layers = [find_target_layer(backbone)]

    with GradCAM(model=cam_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        heatmap = grayscale_cam[0]

    # Grad-CAM utilities require the input image to match heatmap spatial size.
    if input_rgb_float.shape[:2] != heatmap.shape[:2]:
        input_rgb_float = cv2.resize(
            input_rgb_float,
            (heatmap.shape[1], heatmap.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    input_rgb_float = np.clip(input_rgb_float, 0.0, 1.0).astype(np.float32)
    overlay = show_cam_on_image(input_rgb_float, heatmap, use_rgb=True)
    return CamResult(heatmap=heatmap.astype(np.float32), overlay_rgb=overlay.astype(np.uint8))
