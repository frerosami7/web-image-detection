import argparse
import os
from pathlib import Path
import numpy as np
import cv2

try:
    from anomalib.deploy import TorchInferencer
except ImportError as e:  # pragma: no cover
    raise SystemExit("Anomalib not installed. Run: pip install anomalib") from e


def save_artifacts(out_dir: Path, base: str, image_rgb: np.ndarray, mask: np.ndarray, score: float):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize mask to [0,1]
    m = mask.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    heat = (m * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heat_rgb, 0.45, image_rgb, 0.55, 0)

    # Threshold dynamically at 98th percentile
    thr = float(np.quantile(m, 0.98))
    bin_mask = (m >= thr).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed = image_rgb.copy()
    for c in cnts:
        if cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(boxed, (x, y), (x + w, h + y), (255, 0, 0), 2)

    meta_path = out_dir / f"{base}_meta.txt"
    meta_path.write_text(f"score={score:.6f}\nthreshold={thr:.6f}\n")

    cv2.imwrite(str(out_dir / f"{base}_heatmap.png"), cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{base}_mask.png"), bin_mask)
    cv2.imwrite(str(out_dir / f"{base}_boxes.png"), cv2.cvtColor(boxed, cv2.COLOR_RGB2BGR))
def iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            yield p


def infer(args):
    inferencer = TorchInferencer(path=args.torch_model, device="auto")

    out_dir = Path(args.output_dir)
    img_paths = list(iter_images(Path(args.data_root)))
    if not img_paths:
        raise SystemExit(f"No images found under {args.data_root}")

    for idx, ip in enumerate(img_paths):
        image_bgr = cv2.imread(str(ip))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = inferencer.predict(image=image_rgb)

        score = float(getattr(result, "pred_score", 0.0))
        mask = getattr(result, "pred_mask", None)
        if mask is None:
            # create a soft mask from heat_map if available
            heat = getattr(result, "heat_map", None)
            if heat is not None:
                gray = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY) if heat.ndim == 3 else heat
                mask = (gray > np.quantile(gray, 0.98)).astype(np.uint8) * 255
        if mask is None:
            continue
        base = ip.stem
        save_artifacts(out_dir, base, image_rgb, mask, score)

    print(f"Saved artifacts to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run TorchInferencer on images and save visualization artifacts.")
    ap.add_argument("--data-root", required=True, help="Root folder containing images (recursively scanned).")
    ap.add_argument("--torch-model", required=True, help="Path to exported Torch model (.pt)")
    ap.add_argument("--output-dir", default="anomalib_outputs", help="Directory to save artifact images.")
    return ap.parse_args()


if __name__ == "__main__":
    infer(parse_args())
