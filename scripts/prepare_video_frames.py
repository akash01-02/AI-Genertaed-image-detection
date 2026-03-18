from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def _iter_videos(input_dir: Path):
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            d = input_dir / split / cls
            if not d.exists():
                continue
            for p in sorted(d.rglob("*.mp4")):
                yield split, cls, p


def extract_frames(video_path: Path, out_dir: Path, fps: float = 3.0, size: int = 224) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = int(max(1, round(orig_fps / fps))) if orig_fps > 0 else 1

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if i % stride == 0:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"frame_{i:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            count += 1
        i += 1

    cap.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="data/videos")
    parser.add_argument("--output_dir", type=str, required=True, help="data/video_frames")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    videos = list(_iter_videos(input_dir))
    if not videos:
        raise RuntimeError(f"No videos found under: {input_dir}")

    for split, cls, vp in tqdm(videos, desc="Extracting"):
        vid = vp.stem
        out = output_dir / split / cls / vid
        extract_frames(vp, out, fps=args.fps, size=args.size)


if __name__ == "__main__":
    main()
