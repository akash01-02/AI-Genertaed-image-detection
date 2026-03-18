from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aigenerated_detector.data.image_folder import BinaryImageFolderDataset
from aigenerated_detector.data.transforms import build_eval_transforms
from aigenerated_detector.data.video_frames import VideoFramesDataset
from aigenerated_detector.eval.metrics import compute_binary_metrics, probs_to_preds
from aigenerated_detector.models.image_classifier import ImageBinaryClassifier, load_checkpoint
from aigenerated_detector.utils.device import get_device


@torch.inference_mode()
def eval_image(data_dir: Path, checkpoint: Path, model_name: str, image_size: int) -> dict:
    device = get_device(prefer_gpu=True)
    ds = BinaryImageFolderDataset(data_dir, transform=build_eval_transforms(image_size))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = ImageBinaryClassifier(model_name=model_name, pretrained=False).to(device)
    load_checkpoint(model, str(checkpoint), device)
    model.eval()

    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        probs.extend(out.probs.detach().cpu().numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())

    preds = probs_to_preds(np.asarray(probs), threshold=0.5)
    m = compute_binary_metrics(labels, preds.tolist())
    return m.to_dict()


@torch.inference_mode()
def eval_video_frames(data_dir: Path, checkpoint: Path, model_name: str, image_size: int) -> dict:
    device = get_device(prefer_gpu=True)
    ds = VideoFramesDataset(data_dir, transform=build_eval_transforms(image_size), max_frames_per_video=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = ImageBinaryClassifier(model_name=model_name, pretrained=False).to(device)
    load_checkpoint(model, str(checkpoint), device)
    model.eval()

    probs, labels, vids = [], [], []
    for x, y, vid in loader:
        x = x.to(device)
        out = model(x)
        probs.extend(out.probs.detach().cpu().numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())
        vids.extend(list(vid))

    # frame-level
    frame_preds = probs_to_preds(np.asarray(probs), threshold=0.5)
    frame_m = compute_binary_metrics(labels, frame_preds.tolist()).to_dict()

    # video-level
    by_video = defaultdict(list)
    for vid, y, p in zip(vids, labels, probs, strict=True):
        by_video[vid].append((y, p))

    y_true, y_prob = [], []
    for vid, entries in by_video.items():
        ys = [e[0] for e in entries]
        ps = [e[1] for e in entries]
        y_true.append(int(round(float(np.mean(ys)))))
        y_prob.append(float(np.mean(ps)))

    video_preds = probs_to_preds(np.asarray(y_prob), threshold=0.5)
    video_m = compute_binary_metrics(y_true, video_preds.tolist()).to_dict()

    return {"frame_metrics": frame_m, "video_metrics": video_m, "num_videos": len(by_video)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["image", "video"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tf_efficientnet_b0")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)

    if args.task == "image":
        metrics = eval_image(data_dir, checkpoint, args.model_name, args.image_size)
    else:
        metrics = eval_video_frames(data_dir, checkpoint, args.model_name, args.image_size)

    print(metrics)


if __name__ == "__main__":
    main()
