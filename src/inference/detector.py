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

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # OpenCV may be unavailable on some platforms (e.g., Streamlit Cloud)
from PIL import Image, ImageFilter

_COLORMAPS = {}
if cv2 is not None:
    _COLORMAPS = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "HOT": cv2.COLORMAP_HOT,
        "PARULA": getattr(cv2, "COLORMAP_PARULA", getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)),
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
        if cv2 is not None:
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(img)
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.array(pil.resize((w, h), resample))

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
        # Fallback mock: slight blur via PIL if cv2 is unavailable
        if cv2 is not None:
            return cv2.GaussianBlur(x, (5, 5), 0)
        pil = Image.fromarray((x * 255).clip(0, 255).astype(np.uint8))
        pil_blur = pil.filter(ImageFilter.GaussianBlur(radius=1.0))
        return (np.asarray(pil_blur).astype(np.float32)) / 255.0

    def _error_map(self, x: np.ndarray, recon: np.ndarray) -> np.ndarray:
        err = np.mean(np.abs(x - recon), axis=-1)  # (H,W)
        # normalize to [0,1]
        err = (err - err.min()) / (err.max() - err.min() + 1e-8)
        return err.astype(np.float32)

    def _make_heatmap(self, err_norm: np.ndarray, colormap: str) -> np.ndarray:
        heat_uint8 = (err_norm * 255).astype(np.uint8)
        if cv2 is not None:
            cm_code = _COLORMAPS.get(colormap.upper(), getattr(cv2, "COLORMAP_JET", 2))
            heat_bgr = cv2.applyColorMap(heat_uint8, cm_code)
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            return heat_rgb
        # Minimal fallback colormap: red intensity
        r = heat_uint8
        g = (heat_uint8 // 2)
        b = np.zeros_like(heat_uint8)
        return np.stack([r, g, b], axis=-1)

    def _mask_and_boxes(self, err_norm: np.ndarray, threshold: float, min_region_area: int, rotated_boxes: bool = False):
        mask = (err_norm >= threshold).astype(np.uint8) * 255
        if cv2 is None:
            # Pure numpy fallback: connected component labeling to get boxes.
            # Simple 4-neighbor BFS labeling; rotated boxes not supported.
            bin_bool = mask.astype(bool)
            visited = np.zeros_like(bin_bool, dtype=bool)
            h, w = bin_bool.shape
            boxes = []
            def neighbors(y, x):
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        yield ny, nx
            for y in range(h):
                for x in range(w):
                    if bin_bool[y, x] and not visited[y, x]:
                        # BFS component
                        q = [(y,x)]
                        visited[y, x] = True
                        ys = [y]; xs = [x]
                        while q:
                            cy, cx = q.pop()
                            for ny, nx in neighbors(cy, cx):
                                if bin_bool[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    q.append((ny, nx))
                                    ys.append(ny); xs.append(nx)
                        min_y, max_y = min(ys), max(ys)
                        min_x, max_x = min(xs), max(xs)
                        area = (max_x - min_x + 1) * (max_y - min_y + 1)
                        if area >= min_region_area:
                            boxes.append([int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)])
            # Clean mask: keep only accepted boxes
            clean_mask = np.zeros_like(mask)
            for x, y, w_box, h_box in boxes:
                clean_mask[y:y+h_box, x:x+w_box] = 255
            return clean_mask, boxes, []
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []  # axis-aligned [x,y,w,h]
        rboxes = []  # rotated boxes as 4-point polygons
        clean_mask = np.zeros_like(mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_region_area:
                continue
            if rotated_boxes:
                rect = cv2.minAreaRect(cnt)
                pts = cv2.boxPoints(rect)
                pts = np.int32(pts)
                rboxes.append(pts.tolist())
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([int(x), int(y), int(w), int(h)])
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([int(x), int(y), int(w), int(h)])
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        return clean_mask, boxes, rboxes

    def _overlay(self, original: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
        if cv2 is not None:
            return cv2.addWeighted(heatmap, alpha, original, 1 - alpha, 0)
        # Numpy blending fallback
        return (heatmap.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)

    def detect(self, image: np.ndarray,
               threshold: float = 0.5,
               min_region_area: int = 50,
               alpha: float = 0.45,
               colormap: str = "JET",
               dynamic: bool = False,
               dynamic_pct: float = 98.0,
               smooth: bool = False,
               smooth_kernel: int = 5,
               rotated: bool = False) -> Dict[str, Any]:
        """Full artifact generation pipeline for a single RGB uint8 image."""
        orig_h, orig_w = image.shape[:2]
        # prepare input for model
        target_shape = getattr(self.autoencoder, "input_shape", (224, 224, 3))
        x_resized = self._resize(image, target_shape)
        x_norm = self._normalize(x_resized)
        recon = self._reconstruct(x_norm)
        err_norm = self._error_map(x_norm, recon)
        if smooth and smooth_kernel and smooth_kernel % 2 == 1:
            try:
                err_norm = cv2.GaussianBlur(err_norm, (smooth_kernel, smooth_kernel), 0)
                # renormalize after blur
                err_norm = (err_norm - err_norm.min()) / (err_norm.max() - err_norm.min() + 1e-8)
            except Exception:
                pass
        # choose threshold
        thr = float(np.percentile(err_norm, dynamic_pct)) if dynamic else float(threshold)

        heatmap_small = self._make_heatmap(err_norm, colormap)
        mask_small, boxes_small, rboxes_small = self._mask_and_boxes(err_norm, thr, min_region_area, rotated_boxes=rotated)
        # resize artifacts back
        if cv2 is not None:
            heatmap = cv2.resize(heatmap_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            error_map = cv2.resize(err_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap = self._resize(heatmap_small, (orig_h, orig_w, 3))
            mask = self._resize(mask_small, (orig_h, orig_w))
            error_map = self._resize((err_norm * 255).astype(np.uint8), (orig_h, orig_w)).astype(np.float32) / 255.0
        overlay = self._overlay(image, heatmap, alpha)
        # adjust boxes to original scale
        scale_x = orig_w / heatmap_small.shape[1]
        scale_y = orig_h / heatmap_small.shape[0]
        boxes = [[int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)] for x, y, w, h in boxes_small]
        rboxes = []
        if rotated and len(rboxes_small) > 0:
            for pts in rboxes_small:
                pts_np = np.array(pts, dtype=np.float32)
                pts_np[:, 0] *= scale_x
                pts_np[:, 1] *= scale_y
                rboxes.append(pts_np.astype(int).tolist())
        anomaly_score = float(err_norm.mean())
        used_thr = thr
        is_anomaly = anomaly_score >= used_thr  # simple heuristic
        confidence = float(err_norm.max())
        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "confidence": confidence,
            "heatmap": heatmap,
            "mask": mask,
            "overlay": overlay,
            "boxes": boxes,
            "rotated_boxes": rboxes,
            "error_map": error_map,
            "params": {
                "threshold": float(threshold),
                "dynamic": bool(dynamic),
                "dynamic_pct": float(dynamic_pct),
                "min_region_area": min_region_area,
                "alpha": float(alpha),
                "colormap": colormap,
                "smooth": bool(smooth),
                "smooth_kernel": int(smooth_kernel),
                "rotated": bool(rotated),
            },
        }

    # Backward compatibility name
    def detect_anomalies(self, batch: np.ndarray) -> Dict[str, Any]:
        # assume batch shape (1,H,W,C) normalized or not
        img = batch[0]
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return self.detect(img)
