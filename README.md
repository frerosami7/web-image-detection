# Image Anomaly Detection Project

## Overview
This project implements an Image Anomaly Detection system using deep learning techniques, specifically Convolutional Neural Networks (CNN) and Autoencoders. The goal is to identify anomalies in images, which can be useful in various applications such as quality control, medical imaging, and security.

## Features
- **Deep Learning Models**: Utilizes CNN and Autoencoder architectures for effective anomaly detection.
- **Data Preprocessing**: Includes scripts for image preprocessing and augmentation to enhance model robustness.
- **Training and Evaluation**: Comprehensive training and evaluation scripts to ensure model performance.
- **Inference**: Real-time anomaly detection in images and video streams.
- **Web Interface**: User-friendly interfaces built with Gradio and Streamlit for easy interaction.
- **REST API**: Deploys the model as a REST API service for integration with other applications.
- **Real-time Video Processing**: Processes video streams to detect anomalies in real-time.

## Project Structure
```
image-anomaly-detection
├── src
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn_model.py
│   │   └── autoencoder.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── video_processor.py
│   ├── api
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── endpoints.py
│   ├── web_interface
│   │   ├── __init__.py
│   │   ├── gradio_app.py
│   │   └── streamlit_app.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       └── visualization.py
├── configs
│   ├── model_config.yaml
│   └── training_config.yaml
├── notebooks
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
├── tests
│   ├── __init__.py
│   ├── test_models.py
│   └── test_preprocessing.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/enrico310786/Image_Anomaly_Detection.git
   cd Image_Anomaly_Detection
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- To train the model, run:
  ```
  python src/training/train.py
  ```
- To evaluate the model, run:
  ```
  python src/training/evaluate.py
  ```
- To start the web interface using Gradio:
  ```
  python src/web_interface/gradio_app.py
  ```
- To start the web interface using Streamlit:
  ```
  streamlit run src/web_interface/streamlit_app.py
  ```
- To run the REST API service:
  ```
  python src/api/app.py
  ```

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

### Mock vs Real Models
Currently the project uses mock CNN/Autoencoder implementations if TensorFlow is unavailable. The visual pipeline still works, but outputs are random/demonstrative. To enable real anomaly detection:
1. Install a working deep learning backend (TensorFlow or PyTorch).
2. Replace mock model files with trained versions.
3. Ensure the autoencoder's `predict` returns reconstructed images with the same shape as input.

### Programmatic Use of Detector
You can generate artifacts directly:
```python
from src.inference.detector import Detector
from src.models.autoencoder import Autoencoder
import cv2

auto = Autoencoder()  # or load a real trained model
det = Detector(autoencoder=auto)
img = cv2.imread("path/to/image.jpg")[:, :, ::-1]  # BGR to RGB if needed
result = det.detect(img, threshold=0.5, min_region_area=50, alpha=0.45, colormap="JET")
print(result["anomaly_score"], result["boxes"])  # access artifacts
```

Artifacts available in `result`: `anomaly_score`, `is_anomaly`, `confidence`, `heatmap`, `mask`, `overlay`, `boxes`, `error_map`, `params`.

## Results (ReverseDistillation / one_up / v0)
Below are sample outputs produced by the anomaly pipeline (inference example, correct vs incorrect classifications, and confusion matrix). If the images do not render, ensure they are placed in:

`assets/results/ReverseDistillation/one_up/v0/`

| Example inference | Right classification | Bad classification | Confusion matrix |
| ------------------ | ------------------- | ------------------ | ---------------- |
| ![Example inference](assets/results/ReverseDistillation/one_up/v0/example_inference.png) | ![Right classification](assets/results/ReverseDistillation/one_up/v0/plot_right_classification.png) | ![Bad classification](assets/results/ReverseDistillation/one_up/v0/plot_bad_classification.png) | ![Confusion matrix](assets/results/ReverseDistillation/one_up/v0/one_up_confusion_matrix.png) |

To add new experiment results, replicate the folder structure under `assets/results/<Experiment>/<Variant>/<Version>/` and reference them similarly.

## Using Anomalib (Optional Advanced Library)
If you want state-of-the-art anomaly detection instead of the lightweight mock pipeline, you can integrate [Anomalib].

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
- For faster deployment, you can export to OpenVINO: `anomalib export --model Patchcore --ckpt_path path\to\model.ckpt --format openvino --export_root exports/patchcore_ov`.
- Dynamic thresholding (98th percentile of mask intensity) is used in the helper script for adaptive sensitivity.
- Adjust contour area filtering in `anomalib_infer.py` if your anomalies are very small.
 - To use Anomalib inside the Streamlit UI: select "Anomalib" backend, enter the model class (e.g. `Patchcore`) and provide/upload your `.ckpt`. The app will produce heatmap, overlay, mask, and boxes using that checkpoint.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.