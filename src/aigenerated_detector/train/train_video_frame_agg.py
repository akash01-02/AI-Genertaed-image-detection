from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from aigenerated_detector.data.transforms import build_eval_transforms, build_train_transforms
from aigenerated_detector.data.video_frames import VideoFramesDataset
from aigenerated_detector.eval.metrics import compute_binary_metrics, probs_to_preds
from aigenerated_detector.models.image_classifier import ImageBinaryClassifier
from aigenerated_detector.train.training_loop import run_train_val, save_checkpoint
from aigenerated_detector.utils.device import get_device
from aigenerated_detector.utils.io import ensure_dir, save_json
from aigenerated_detector.utils.seed import set_seed


def _plot_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    arr = np.array(cm)
    plt.figure(figsize=(4, 4))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _video_level_metrics(video_ids: list[str], labels: list[int], probs: list[float]) -> dict:
    by_video: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for vid, y, p in zip(video_ids, labels, probs, strict=True):
        by_video[vid].append((y, p))

    y_true: list[int] = []
    y_prob: list[float] = []
    for vid, entries in by_video.items():
        ys = [e[0] for e in entries]
        ps = [e[1] for e in entries]
        # assume consistent label within a video
        y_true.append(int(round(float(np.mean(ys)))))
        y_prob.append(float(np.mean(ps)))

    preds = probs_to_preds(np.asarray(y_prob), threshold=0.5)
    m = compute_binary_metrics(y_true, preds.tolist())
    return {"num_videos": len(by_video), "metrics": m.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Root with train/val/test subfolders")
    parser.add_argument("--model_name", type=str, default="tf_efficientnet_b0")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_frames_per_video", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=True)

    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.output_dir)

    train_ds = VideoFramesDataset(
        data_dir / "train",
        transform=build_train_transforms(args.image_size),
        max_frames_per_video=args.max_frames_per_video,
    )
    val_ds = VideoFramesDataset(
        data_dir / "val",
        transform=build_eval_transforms(args.image_size),
        max_frames_per_video=args.max_frames_per_video,
    )
    test_ds = VideoFramesDataset(
        data_dir / "test",
        transform=build_eval_transforms(args.image_size),
        max_frames_per_video=args.max_frames_per_video,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = ImageBinaryClassifier(model_name=args.model_name, pretrained=True).to(device)

    history, _best_epoch = run_train_val(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Evaluate on test: frame-level + video-level
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    video_ids: list[str] = []
    with torch.inference_mode():
        for x, y, vid in test_loader:
            x = x.to(device)
            out = model(x)
            probs.extend(out.probs.detach().cpu().numpy().tolist())
            labels.extend(y.numpy().astype(int).tolist())
            video_ids.extend(list(vid))

    frame_preds = probs_to_preds(np.asarray(probs), threshold=0.5)
    frame_metrics = compute_binary_metrics(labels, frame_preds.tolist())
    video_metrics = _video_level_metrics(video_ids, labels, probs)

    save_checkpoint(model, out_dir / "best.pt")
    save_json(
        {
            "history": history,
            "test_frame_metrics": frame_metrics.to_dict(),
            "test_video_metrics": video_metrics,
            "model_name": args.model_name,
            "image_size": args.image_size,
        },
        out_dir / "metrics.json",
    )
    _plot_confusion_matrix(frame_metrics.confusion_matrix, out_dir / "confusion_matrix_frames.png")
    _plot_confusion_matrix(video_metrics["metrics"]["confusion_matrix"], out_dir / "confusion_matrix_videos.png")

    print("Saved:", str(out_dir / "best.pt"))
    print("Test frame metrics:", frame_metrics.to_dict())
    print("Test video metrics:", video_metrics)


if __name__ == "__main__":
    main()
