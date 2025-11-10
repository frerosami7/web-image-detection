import numpy as np
from typing import Dict, Any

try:
    import cv2  # for better colormap & boxes
except Exception:
    cv2 = None

try:
    from anomalib.deploy import TorchInferencer
except ImportError as e:  # pragma: no cover
    TorchInferencer = None  # type: ignore


def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img


def single_image_predict(
    image_rgb: np.ndarray,
    model_name: str = "",  # unused with TorchInferencer, kept for backward-compat
    ckpt_path: str = "",   # unused; preserved for API compatibility
    image_size: int = 256,
    threshold_mode: str = "dynamic",
    dynamic_pct: float = 98.0,
    static_threshold: float = 0.5,
    min_region_area: int = 50,
    alpha: float = 0.45,
    colormap: str = "JET",
    rotated: bool = False,
    smooth: bool = False,
    smooth_kernel: int = 5,
    torch_model_path: str = "",
    inferencer: Any = None,
) -> Dict[str, Any]:
    if TorchInferencer is None:
        raise RuntimeError("Anomalib not installed. Run: pip install anomalib")

    if not torch_model_path:
        raise ValueError("torch_model_path is required for TorchInferencer.")

    # Optional resize before inference to control input size similar to external reference
    img = image_rgb
    if isinstance(image_size, int) and image_size > 0 and (img.shape[0] != image_size or img.shape[1] != image_size):
        if cv2 is not None:
            img = cv2.resize(img, (int(image_size), int(image_size)))
        else:
            # Fallback resize via numpy/PIL
            from PIL import Image as PILImage
            img = np.array(PILImage.fromarray(img).resize((int(image_size), int(image_size))))

    # Run inference (use cached inferencer if provided)
    if inferencer is None:
        inferencer = TorchInferencer(path=torch_model_path, device="auto")
    result = inferencer.predict(image=img)

    # Extract fields robustly
    pred_score = float(getattr(result, "pred_score", 0.0))
    pred_label = int(getattr(result, "pred_label", 0))
    pred_mask = getattr(result, "pred_mask", None)
    heat_map = getattr(result, "heat_map", None)
    out_image = getattr(result, "image", img)

    # Normalize/prepare mask
    m = None
    bin_mask = None
    if pred_mask is not None:
        m = pred_mask.astype(np.float32)
        # Some inferencers return 0/255 masks; normalize to [0,1]
        if m.max() > 1.0:
            m = m / 255.0
        if smooth and smooth_kernel % 2 == 1 and cv2 is not None:
            m = cv2.GaussianBlur(m, (int(smooth_kernel), int(smooth_kernel)), 0)
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)

        # Threshold selection
        if threshold_mode == "dynamic":
            thr = float(np.quantile(m, dynamic_pct / 100.0))
        else:
            thr = float(static_threshold)
        bin_mask = (m >= thr).astype(np.uint8) * 255
    else:
        thr = float(static_threshold) if threshold_mode == "static" else 0.5

    # Heatmap: prefer inferencer heat_map; fallback to colormap from mask
    heat_rgb = None
    if heat_map is not None:
        heat_rgb = _ensure_uint8_rgb(heat_map)
    elif m is not None:
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
    base_img = _ensure_uint8_rgb(out_image)
    overlay = None
    if heat_rgb is not None:
        overlay = (heat_rgb.astype(np.float32) * alpha + base_img.astype(np.float32) * (1 - alpha)).clip(0, 255).astype(np.uint8)
    else:
        # Classification-only fallback: show original image as overlay to avoid missing key warnings
        overlay = base_img

    # Bounding boxes
    boxes = []
    rboxes = []
    if bin_mask is not None:
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

    # Assemble safe result
    classification_only = bool(heat_rgb is None and bin_mask is None)
    out = {
        "anomaly_score": float(pred_score),
        "is_anomaly": bool(pred_label == 1),
        "confidence": float(m.max()) if m is not None else float(pred_score),
        "heatmap": heat_rgb,
        "mask": bin_mask,
        "overlay": overlay,
        "boxes": boxes,
        "rotated_boxes": rboxes,
        "error_map": m,
        "params": {
            "backend": "Anomalib-TorchInferencer",
            "model": model_name or "embedded",
            "torch_model_path": torch_model_path,
            "threshold": thr,
            "threshold_mode": threshold_mode,
            "dynamic_pct": dynamic_pct,
            "classification_only": classification_only,
        },
    }

    # Remove keys with None to simplify UI guards
    return {k: v for k, v in out.items() if v is not None}
