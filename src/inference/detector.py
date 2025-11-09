"""Unified Detector that produces visual anomaly artifacts.

This implementation works with a (mock) autoencoder reconstruction model. It returns:
    - anomaly_score: mean reconstruction error
    - heatmap: RGB heatmap image
    - mask: binary uint8 mask (255 anomalous)
    - overlay: original image blended with heatmap
    - boxes: list of [x,y,w,h] for connected anomalous regions
    - error_map: float32 map in [0,1]

Parameters supported:
    threshold (float): threshold on normalized error map for mask
    min_region_area (int): minimum contour area to keep
    alpha (float): overlay blend factor
    colormap (str): OpenCV colormap name (JET, TURBO, HOT, PARULA)
"""

from typing import Dict, Any, List
import numpy as np
import cv2

_COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "PARULA": cv2.COLORMAP_PARULA if hasattr(cv2, "COLORMAP_PARULA") else cv2.COLORMAP_JET,
}


class Detector:
    def __init__(self, autoencoder=None):
        self.autoencoder = autoencoder  # expects .input_shape and .predict(batch)

    # Backward compatibility simple predict returning anomalous / normal.
    def predict(self, batch: np.ndarray) -> int:
        result = self.detect(batch[0])  # assume first image in batch
        return 1 if result["is_anomaly"] else 0

    def _resize(self, img: np.ndarray, shape) -> np.ndarray:
        h, w = shape[:2]
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0

    def _reconstruct(self, x: np.ndarray) -> np.ndarray:
        # x expected shape (H,W,C) normalized
        if self.autoencoder and hasattr(self.autoencoder, "predict"):
            try:
                recon = self.autoencoder.predict(x[None, ...])[0]
                recon = recon.astype(np.float32)
                if recon.shape != x.shape:
                    recon = self._resize(recon, x.shape)
                return recon
            except Exception:
                pass
        # Fallback mock: slight Gaussian blur acts as reconstruction
        recon_mock = cv2.GaussianBlur(x, (5, 5), 0)
        return recon_mock

    def _error_map(self, x: np.ndarray, recon: np.ndarray) -> np.ndarray:
        err = np.mean(np.abs(x - recon), axis=-1)  # (H,W)
        # normalize to [0,1]
        err = (err - err.min()) / (err.max() - err.min() + 1e-8)
        return err.astype(np.float32)

    def _make_heatmap(self, err_norm: np.ndarray, colormap: str) -> np.ndarray:
        cm_code = _COLORMAPS.get(colormap.upper(), cv2.COLORMAP_JET)
        heat_uint8 = (err_norm * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_uint8, cm_code)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        return heat_rgb

    def _mask_and_boxes(self, err_norm: np.ndarray, threshold: float, min_region_area: int) -> (np.ndarray, List[List[int]]):
        mask = (err_norm >= threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        clean_mask = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_region_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([int(x), int(y), int(w), int(h)])
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        return clean_mask, boxes

    def _overlay(self, original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
        return cv2.addWeighted(heatmap, alpha, original, 1 - alpha, 0)

    def detect(self, image: np.ndarray, threshold: float = 0.5, min_region_area: int = 50,
               alpha: float = 0.45, colormap: str = "JET") -> Dict[str, Any]:
        """Full artifact generation pipeline for a single RGB uint8 image."""
        orig_h, orig_w = image.shape[:2]
        # prepare input for model
        target_shape = getattr(self.autoencoder, "input_shape", (224, 224, 3))
        x_resized = self._resize(image, target_shape)
        x_norm = self._normalize(x_resized)
        recon = self._reconstruct(x_norm)
        err_norm = self._error_map(x_norm, recon)
        heatmap_small = self._make_heatmap(err_norm, colormap)
        mask_small, boxes_small = self._mask_and_boxes(err_norm, threshold, min_region_area)
        # resize artifacts back
        heatmap = cv2.resize(heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        overlay = self._overlay(image, heatmap, alpha)
        # scale error map back
        error_map = cv2.resize(err_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # adjust boxes to original scale
        scale_x = orig_w / heatmap_small.shape[1]
        scale_y = orig_h / heatmap_small.shape[0]
        boxes = [[int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)] for x, y, w, h in boxes_small]
        anomaly_score = float(err_norm.mean())
        is_anomaly = anomaly_score >= threshold  # simple heuristic
        confidence = float(err_norm.max())
        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "heatmap": heatmap,
            "mask": mask,
            "overlay": overlay,
            "boxes": boxes,
            "error_map": error_map,
            "params": {
                "threshold": threshold,
                "min_region_area": min_region_area,
                "alpha": alpha,
                "colormap": colormap,
            },
        }

    # Backward compatibility name
    def detect_anomalies(self, batch: np.ndarray) -> Dict[str, Any]:
        # assume batch shape (1,H,W,C) normalized or not
        img = batch[0]
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return self.detect(img)
