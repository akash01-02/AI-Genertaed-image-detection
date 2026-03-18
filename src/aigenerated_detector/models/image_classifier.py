from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor  # shape [B]
    probs: torch.Tensor  # shape [B]


class ImageBinaryClassifier(nn.Module):
    """Transfer-learning wrapper for binary classification (real=0, fake=1)."""

    def __init__(self, model_name: str = "tf_efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        logits = self.backbone(x).squeeze(1)
        probs = torch.sigmoid(logits)
        return ModelOutput(logits=logits, probs=probs)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model
