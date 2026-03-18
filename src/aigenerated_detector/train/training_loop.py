from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochResult:
    loss: float
    probs: list[float]
    labels: list[int]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_probs: list[float] = []
    all_labels: list[int] = []

    criterion = nn.BCEWithLogitsLoss()
    pbar = tqdm(loader, desc="train" if is_train else "eval", leave=False)

    for batch in pbar:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, _ = batch
        x = x.to(device)
        y = y.to(device).float()

        out = model(x)
        logits = out.logits
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        all_probs.extend(out.probs.detach().cpu().numpy().tolist())
        all_labels.extend(y.detach().cpu().int().numpy().tolist())
        pbar.set_postfix(loss=float(loss.item()))

    n = len(loader.dataset)
    return EpochResult(loss=total_loss / max(n, 1), probs=all_probs, labels=all_labels)


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, str(path))


def run_train_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> tuple[list[dict], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict] = []
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_res = _run_epoch(model, train_loader, device, optimizer)
        val_res = _run_epoch(model, val_loader, device, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_res.loss,
            "val_loss": val_res.loss,
        }
        history.append(row)

        if val_res.loss < best_val_loss:
            best_val_loss = val_res.loss
            best_epoch = epoch

    return history, best_epoch
