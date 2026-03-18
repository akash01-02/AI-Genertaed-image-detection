# AI-Generated Image & Deepfake Video Detection System with Explainability

A practical, laptop-friendly deepfake detector:

- **Images**: CNN transfer learning (EfficientNet / ResNet).
- **Videos**: extract frames → run the same CNN per-frame → **aggregate** frame probabilities into a video score (lightweight and fast on CPU).
- **Explainability**: Grad-CAM heatmaps for images and sampled video frames.
- **UI**: Streamlit app to upload images/videos and view results + heatmaps.

This repo is designed to be **runnable on a normal laptop** (CPU-first), and provides training, evaluation (accuracy/precision/recall/F1/confusion matrix), and inference.

## 1) Setup

### Prereqs
- Python **3.10+** recommended
- (Optional) NVIDIA GPU + CUDA for faster training

### Install

```bash
cd d:\Projects\FakePhotoDetection
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 2) Datasets (publicly available)

You can use any *real vs fake* face dataset for images, and any deepfake dataset for videos.

### Image datasets (examples)
- **GAN vs Real** datasets on Kaggle / open datasets (e.g., “real and fake face detection” style datasets).
- Any dataset that can be arranged into a **folder-per-class** structure.

### Video datasets (examples)
- **FaceForensics++**, **Celeb-DF**, **DFDC** (DeepFake Detection Challenge)

Notes:
- Some datasets require registration/approval (FaceForensics++). Follow their official access instructions.
- This project expects you to prepare **frames** from videos for training, or you can perform on-the-fly extraction for inference.

## 3) Expected folder structure

Create this structure under `data/`:

### For image training

```
data/images/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

### For video frame training (optional fine-tune)

```
data/video_frames/
  train/
    real/<video_id>/*.jpg
    fake/<video_id>/*.jpg
  val/
    real/<video_id>/*.jpg
    fake/<video_id>/*.jpg
  test/
    real/<video_id>/*.jpg
    fake/<video_id>/*.jpg
```

To prepare frames from videos into the above format, use:

```bash
python scripts/prepare_video_frames.py --input_dir data/videos --output_dir data/video_frames --fps 3 --size 224
```

`data/videos` should be organized as:

```
data/videos/
  train/real/*.mp4
  train/fake/*.mp4
  val/real/*.mp4
  val/fake/*.mp4
  test/real/*.mp4
  test/fake/*.mp4
```

## 4) Train

### Train image model (transfer learning)

```bash
python -m aigenerated_detector.train.train_image \
  --data_dir data/images \
  --model_name tf_efficientnet_b0 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-4 \
  --output_dir outputs/image_run
```

This writes:
- `outputs/image_run/best.pt` (best checkpoint)
- `outputs/image_run/metrics.json`
- `outputs/image_run/confusion_matrix.png`

### Train/fine-tune on video frames (optional)

```bash
python -m aigenerated_detector.train.train_video_frame_agg \
  --data_dir data/video_frames \
  --model_name tf_efficientnet_b0 \
  --epochs 3 \
  --batch_size 16 \
  --lr 3e-4 \
  --output_dir outputs/video_frame_run
```

## 5) Evaluate

```bash
python scripts/evaluate_checkpoint.py \
  --task image \
  --data_dir data/images/test \
  --checkpoint outputs/image_run/best.pt
```

For video-frame evaluation with video-level aggregation:

```bash
python scripts/evaluate_checkpoint.py \
  --task video \
  --data_dir data/video_frames/test \
  --checkpoint outputs/video_frame_run/best.pt
```

## 6) Run Streamlit app

```bash
streamlit run streamlit_app/app.py
```

In the UI:
- Upload an **image** or **video**.
- Select a checkpoint.
- See label + confidence and Grad-CAM heatmaps.

## 7) Labels and confidence

We predict a **fake/AI probability** $p_{fake}$.

- $p_{fake} \ge 0.70$ → **Likely AI-Generated**
- $0.40 \le p_{fake} < 0.70$ → **Suspicious**
- $p_{fake} < 0.40$ → **Likely Real**

Confidence is shown as:
- For **Likely AI-Generated**: $p_{fake}$
- For **Likely Real**: $1 - p_{fake}$
- For **Suspicious**: closeness to 0.5 (uncertain region)

## 8) Resume bullets

- Built an end-to-end deepfake detection system for images and videos using EfficientNet transfer learning, with frame-wise inference and aggregation for laptop-friendly video scoring.
- Implemented Grad-CAM explainability to visualize manipulated regions on images and sampled video frames, improving model transparency for end users.
- Delivered a Streamlit application enabling real-time upload, inference, confidence scoring, and heatmap visualization.
- Evaluated models with accuracy, precision, recall, F1-score, and confusion matrix; added reproducible training scripts and dataset preparation utilities.

## 9) Project abstract

This project detects AI-generated images and deepfake videos using transfer-learned convolutional networks. For videos, frames are sampled and evaluated by the same CNN, and frame probabilities are aggregated into a video-level score for efficient CPU inference. Grad-CAM explainability provides localized visual evidence supporting predictions. A Streamlit interface enables simple upload-and-analyze workflows and displays predictions with confidence and heatmaps.

## 10) Future improvements

- Replace frame aggregation with a temporal model (CNN + LSTM/Transformer) when GPU compute is available.
- Add face detection/alignment (RetinaFace/MTCNN) and track faces across frames to reduce background leakage.
- Calibrate probabilities (temperature scaling) and add per-dataset threshold tuning.
- Add adversarial robustness checks and cross-dataset generalization benchmarks.
- Provide model cards and bias analysis across demographics.
