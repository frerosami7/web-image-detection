import argparse
import os
from pathlib import Path
import numpy as np
import cv2

try:
    from anomalib.data import Folder
    from anomalib.models import Patchcore
    from anomalib.engine import Engine
except ImportError as e:  # pragma: no cover
    raise SystemExit("Anomalib not installed. Run: pip install anomalib") from e


def load_engine(model_name: str = "Patchcore"):
    # Extendable: map names to classes
    model_cls = getattr(__import__("anomalib.models", fromlist=[model_name]), model_name)
    model = model_cls()
    engine = Engine()
    return engine, model


def build_datamodule(data_root: str, image_size: int = 256):
    return Folder(root=data_root, task="segmentation", image_size=image_size)


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


def infer(args):
    engine, model = load_engine(args.model)
    dm = build_datamodule(args.data_root, args.image_size)

    preds = engine.predict(datamodule=dm, model=model, ckpt_path=args.ckpt, return_predictions=True)
    for idx, p in enumerate(preds):
        # robust extraction
        img = p.get("image") if isinstance(p, dict) else getattr(p, "image")
        score = float(p.get("pred_scores", p.get("pred_score", 0.0))) if isinstance(p, dict) else float(getattr(p, "pred_score", 0.0))
        mask = p.get("pred_masks") if isinstance(p, dict) else getattr(p, "pred_mask", None)
        if mask is None:
            continue
        base = f"sample_{idx:04d}"
        save_artifacts(Path(args.output_dir), base, img, mask, score)

    print(f"Saved artifacts to {args.output_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run anomalib model inference and save visualization artifacts.")
    ap.add_argument("--data-root", required=True, help="Root folder containing images (train/test style not required for inference).")
    ap.add_argument("--ckpt", required=False, default=None, help="Path to model checkpoint .ckpt (if omitted, uses random init).")
    ap.add_argument("--model", default="Patchcore", help="Model class name in anomalib.models (default Patchcore).")
    ap.add_argument("--image-size", type=int, default=256, help="Image resize for datamodule.")
    ap.add_argument("--output-dir", default="anomalib_outputs", help="Directory to save artifact images.")
    return ap.parse_args()


if __name__ == "__main__":
    infer(parse_args())
