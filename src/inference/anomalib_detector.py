import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any

try:
    import cv2  # for better colormap & boxes
except Exception:
    cv2 = None

try:
    from anomalib.engine import Engine
except ImportError:
    Engine = None  # type: ignore
try:
    from anomalib.data import Folder
except ImportError:
    Folder = None  # type: ignore


def load_model(model_name: str):
    mod = __import__("anomalib.models", fromlist=[model_name])
    ModelCls = getattr(mod, model_name)
    return ModelCls()


def single_image_predict(image_rgb: np.ndarray,
                         model_name: str,
                         ckpt_path: str,
                         image_size: int = 256,
                         threshold_mode: str = "dynamic",
                         dynamic_pct: float = 98.0,
                         static_threshold: float = 0.5,
                         min_region_area: int = 50,
                         alpha: float = 0.45,
                         colormap: str = "JET",
                         rotated: bool = False,
                         smooth: bool = False,
                         smooth_kernel: int = 5) -> Dict[str, Any]:
    if Engine is None or Folder is None:
        raise RuntimeError("Anomalib not installed. Run: pip install anomalib")

    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "sample.png"
        import PIL.Image as PImage
        PImage.fromarray(image_rgb).save(img_path)
        dm = Folder(root=str(Path(td)), task="segmentation", image_size=int(image_size))
        engine = Engine()
        model = load_model(model_name)
        preds = engine.predict(datamodule=dm, model=model, ckpt_path=str(ckpt_path), return_predictions=True)
    if not preds:
        raise RuntimeError("No predictions returned by Anomalib.")
    p = preds[0]
    # robust extraction
    img = p.get("image") if isinstance(p, dict) else getattr(p, "image")
    score = float(p.get("pred_scores", p.get("pred_score", 0.0))) if isinstance(p, dict) else float(getattr(p, "pred_score", 0.0))
    mask = p.get("pred_masks") if isinstance(p, dict) else getattr(p, "pred_mask", None)
    if mask is None:
        raise RuntimeError("Model did not return a mask. Choose a segmentation-capable model.")

    m = mask.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    if smooth and smooth_kernel % 2 == 1 and cv2 is not None:
        m = cv2.GaussianBlur(m, (smooth_kernel, smooth_kernel), 0)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)

    # Threshold selection
    if threshold_mode == "dynamic":
        thr = float(np.quantile(m, dynamic_pct / 100.0))
    else:
        thr = float(static_threshold)
    bin_mask = (m >= thr).astype(np.uint8) * 255

    # Colormap
    if cv2 is not None:
        cmap_code = {
            "JET": getattr(cv2, "COLORMAP_JET", 2),
            "TURBO": getattr(cv2, "COLORMAP_TURBO", 20),
            "HOT": getattr(cv2, "COLORMAP_HOT", 11),
            "PARULA": getattr(cv2, "COLORMAP_PARULA", getattr(cv2, "COLORMAP_TURBO", 20)),
        }.get(colormap.upper(), getattr(cv2, "COLORMAP_JET", 2))
        heat_bgr = cv2.applyColorMap((m * 255).astype(np.uint8), cmap_code)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    else:
        heat_u = (m * 255).astype(np.uint8)
        heat_rgb = np.stack([heat_u, heat_u // 2, np.zeros_like(heat_u)], axis=-1)

    # Overlay
    overlay = (heat_rgb.astype(np.float32) * alpha + image_rgb.astype(np.float32) * (1 - alpha)).clip(0, 255).astype(np.uint8)

    boxes = []
    rboxes = []
    if cv2 is not None:
        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_region_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append([int(x), int(y), int(w), int(h)])
            if rotated:
                rect = cv2.minAreaRect(c)
                pts = cv2.boxPoints(rect)
                rboxes.append([(int(px), int(py)) for px, py in pts])
    else:
        # Simple numpy BFS fallback
        bin_bool = bin_mask.astype(bool)
        visited = np.zeros_like(bin_bool, dtype=bool)
        h, w = bin_bool.shape
        def nb(y, x):
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    yield ny, nx
        for y in range(h):
            for x in range(w):
                if bin_bool[y, x] and not visited[y, x]:
                    q = [(y,x)]
                    visited[y, x] = True
                    ys = [y]; xs = [x]
                    while q:
                        cy, cx = q.pop()
                        for ny, nx in nb(cy, cx):
                            if bin_bool[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                q.append((ny, nx))
                                ys.append(ny); xs.append(nx)
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    area = (max_x - min_x + 1) * (max_y - min_y + 1)
                    if area >= min_region_area:
                        boxes.append([int(min_x), int(min_y), int(max_x - min_x + 1), int(max_y - min_y + 1)])

    result = {
        "anomaly_score": float(score),
        "is_anomaly": bool(score >= 0.5),
        "confidence": float(m.max()),
        "heatmap": heat_rgb,
        "mask": bin_mask,
        "overlay": overlay,
        "boxes": boxes,
        "rotated_boxes": rboxes,
        "error_map": m,
        "params": {
            "backend": "Anomalib",
            "model": model_name,
            "ckpt_path": ckpt_path,
            "threshold": thr,
            "threshold_mode": threshold_mode,
            "dynamic_pct": dynamic_pct,
        },
    }
    return result
