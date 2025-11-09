# Image Anomaly Detection Project

## Overview
This project now focuses on visual anomaly detection using **Anomalib** (state-of-the-art PyTorch-based anomaly detection library). Earlier mock CNN/Autoencoder components have been removed. You can train an Anomalib model on a folder dataset and run real inference through the Streamlit UI or CLI scripts.

## Features
- **Anomalib Models**: Patchcore, Padim, STFPM and others via a unified API.
- **Folder Dataset Training**: Train directly on a normal/anomalous image folder layout.
- **Artifact Generation**: Heatmap, mask, overlay, bounding boxes and anomaly score.
- **Adaptive Thresholding**: Static or dynamic percentile-based mask creation.
- **Streamlit UI**: Upload an image + checkpoint for instant visualization.
- **CLI Inference Script**: Batch process a directory with artifact export.
- **Optional REST/API**: Existing endpoints can be adapted to serve Anomalib predictions.

## Simplified Structure (Key Files)
```
src/
  inference/
    anomalib_detector.py     <- single-image Anomalib inference & artifacts
    detector.py              <- legacy reconstruction-based pipeline (fallback/no ckpt)
  training/
    train_anomalib.py        <- simple folder-based training script
  web_interface/
    streamlit_app.py         <- Anomalib-only UI (upload image & ckpt)
  data/preprocessing.py      <- basic preprocessing utilities
requirements.txt             <- includes anomalib, torch; tensorflow removed
README.md                    <- this file
runtime.txt                  <- pins Python version (e.g., 3.10.13 for deployment)
```

## Installation
```powershell
git clone https://github.com/frerosami7/web-image-detection.git
cd web-image-detection
python -m venv .venv
./.venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
```

## Core Workflows

### 1. Prepare Folder Dataset
Required layout (segmentation or classification tasks both supported):
```
data/<category>/
  train/
    good/           # only normal images
  test/
    good/
    anomalous/      # anomalous samples
  ground_truth/     # optional pixel masks (segmentation)
```

### 2. Train Anomalib Model
```powershell
python src/training/train_anomalib.py --data-root data/my_category --model Patchcore --task segmentation --image-size 256
```
Checkpoint (.ckpt) will be saved under Anomalib's default results path (varies by version). Locate it via:
```powershell
Get-ChildItem -Recurse -Filter *.ckpt | Select-Object FullName
```

### 3. Streamlit Inference (Single Image)
```powershell
streamlit run src/web_interface/streamlit_app.py
```
In the sidebar:
1. Enter model class (e.g., Patchcore).
2. Provide or upload checkpoint (.ckpt).
3. Optionally adjust threshold mode, percentile, smoothing, region area.
4. Upload an image and click "Detect Anomalies".

### 4. Batch Inference CLI
```powershell
python src/inference/anomalib_infer.py --data-root path\to\images --ckpt path\to\model.ckpt --model Patchcore --output-dir anomalib_outputs
```
Generates: `*_heatmap.png`, `*_overlay.png`, `*_mask.png`, `*_boxes.png`, `*_meta.txt`.

### Visual Anomaly Artifacts (Streamlit)
The Streamlit interface now produces several visual artifacts to help interpret anomalies:

Artifact | Description
-------- | -----------
Anomaly Score | Mean normalized reconstruction error (higher = more anomalous)
Heatmap | Pixel-wise error intensity rendered with a selectable colormap
Binary Mask | Thresholded heatmap (configurable `Mask threshold`)
Overlay | Original image blended with the heatmap (`Overlay alpha`)
Bounding Boxes | Connected anomalous regions above `Min region area`

You can tune these parameters live in the sidebar:

Parameter | Purpose
--------- | -------
Mask threshold | Controls sensitivity of the binary mask (0 = permissive, 1 = strict)
Min region area | Filters out tiny noisy regions below the given pixel area
Overlay alpha | Blend strength of heatmap over the original image
Colormap | Color style for the heatmap (JET, TURBO, HOT, PARULA)

### Legacy Reconstruction Detector
File `src/inference/detector.py` remains as a lightweight, mock-compatible reconstruction+error demo. For production use, prefer `anomalib_detector.py` via Streamlit or the batch script.

### Programmatic Single Image (Anomalib)
```python
import numpy as np
from PIL import Image
from src.inference.anomalib_detector import single_image_predict

image_rgb = np.array(Image.open("sample.png").convert("RGB"))
result = single_image_predict(
  image_rgb=image_rgb,
  model_name="Patchcore",
  ckpt_path="path/to/best.ckpt",
  threshold_mode="dynamic",
  dynamic_pct=98.0,
)
print(result["anomaly_score"], result["boxes"])  # access artifacts
```
Result keys: `anomaly_score`, `is_anomaly`, `confidence`, `heatmap`, `mask`, `overlay`, `boxes`, `rotated_boxes`, `error_map`, `params`.

## Results (ReverseDistillation / one_up / v0)
Below are sample outputs produced by the anomaly pipeline (inference example, correct vs incorrect classifications, and confusion matrix). If the images do not render, ensure they are placed in:

`assets/results/ReverseDistillation/one_up/v0/`

| Example inference | Right classification | Bad classification | Confusion matrix |
| ------------------ | ------------------- | ------------------ | ---------------- |
| ![Example inference](assets/results/ReverseDistillation/one_up/v0/example_inference.png) | ![Right classification](assets/results/ReverseDistillation/one_up/v0/plot_right_classification.png) | ![Bad classification](assets/results/ReverseDistillation/one_up/v0/plot_bad_classification.png) | ![Confusion matrix](assets/results/ReverseDistillation/one_up/v0/one_up_confusion_matrix.png) |

To add new experiment results, replicate the folder structure under `assets/results/<Experiment>/<Variant>/<Version>/` and reference them similarly.

## Anomalib Quick Reference

### Install
```powershell
pip install anomalib
```

### Train (Example: Patchcore on MVTec "transistor")
```powershell
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category transistor
```

### Inference & Localization (CLI)
```powershell
anomalib predict --model anomalib.models.Patchcore --data anomalib.data.MVTecAD --data.category transistor --ckpt_path path\to\model.ckpt --return_predictions
```

### Scripted Inference With Artifacts
Use `src/inference/anomalib_infer.py` to process a folder of images and save heatmap/mask/overlay/boxes.
```powershell
python src/inference/anomalib_infer.py --data-root path\to\images --ckpt path\to\model.ckpt --model Patchcore --output-dir anomalib_outputs
```

Artifacts are saved per sample: `_heatmap.png`, `_overlay.png`, `_mask.png`, `_boxes.png`, plus a `_meta.txt` with score and threshold.

### Notes
- Export to OpenVINO for speed: `anomalib export --model Patchcore --ckpt_path path\to\model.ckpt --format openvino --export_root exports/patchcore_ov`.
- Dynamic thresholding (percentile) improves robustness across heterogeneous samples.
- Adjust `--image-size` for a trade-off between speed and localization granularity.
- Rotated boxes require OpenCV; on environments without libGL they gracefully degrade to axis-aligned approximations.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.