import streamlit as st
import numpy as np
import sys
from pathlib import Path
import inspect
import tempfile

# Ensure project root on sys.path for Streamlit Cloud (where working dir may differ)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import cv2
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = str(e)
from PIL import Image

try:
    from src.models.autoencoder import Autoencoder
    from src.inference.detector import Detector
except ModuleNotFoundError:
    # Fallback: attempt relative import if src package resolution fails
    sys.stderr.write("[WARN] Falling back to relative imports; ensure src/__init__.py exists.\n")
    from models.autoencoder import Autoencoder  # type: ignore
    from inference.detector import Detector  # type: ignore

st.set_page_config(page_title="Image Anomaly Detection", layout="wide")

# Initialize models (mock/built-in)
autoencoder = Autoencoder()
detector = Detector(autoencoder=autoencoder)

st.title("Image Anomaly Detection")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    # Convert PIL to numpy RGB uint8
    image_array = np.array(image.convert("RGB"))

    # Parameters sidebar
    with st.sidebar:
        st.header("Backend")
        backend = st.radio("Choose engine", ["Built-in (Mock)", "Anomalib"], index=0, help="Use Anomalib if you have a trained checkpoint.")

        st.header("Parameters")
        mode = st.selectbox("Threshold mode", ["Static", "Dynamic (percentile)"])
        if mode == "Static":
            threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01, help="Threshold on normalized error map")
            dynamic = False
            dynamic_pct = 98.0
        else:
            threshold = 0.5  # unused in dynamic
            dynamic = True
            dynamic_pct = st.slider("Dynamic percentile", 80.0, 99.9, 98.0, 0.1,
                                     help="Use percentile of error map as adaptive threshold")
        min_region_area = st.number_input("Min region area (px)", min_value=0, value=50, step=5,
                                          help="Small regions are removed")
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.45, 0.01, help="Blend factor for heatmap overlay")
        smooth = st.checkbox("Reduce noise (smooth)", value=False)
        smooth_kernel = st.selectbox("Smooth kernel", [3,5,7], index=1)
        draw_rotated = st.checkbox("Rotated boxes", value=False, help="Use minimum-area rotated rectangles")
        colormap = st.selectbox("Colormap", ["JET", "TURBO", "HOT", "PARULA"], index=0)

        if backend == "Anomalib":
            st.divider()
            st.subheader("Anomalib Settings")
            model_name = st.text_input("Model class", value="Patchcore", help="Class in anomalib.models, e.g., Patchcore, Padim, Stfpm")
            image_size = st.number_input("Image size", min_value=64, max_value=1024, value=256, step=32)
            ckpt_path_text = st.text_input("Checkpoint path (.ckpt)", value="", help="Absolute or repo-relative path on server")
            ckpt_upload = st.file_uploader("Or upload checkpoint", type=["ckpt"], accept_multiple_files=False)

    # Detect anomalies
    if cv2 is None:
        st.error("OpenCV failed to import on this platform. Details: " + CV2_IMPORT_ERROR)
    elif st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner('Analyzing image...'):
            try:
                if st.session_state.get("backend_choice"):
                    pass  # placeholder to keep streamlit stable on reruns

                if backend == "Anomalib":
                    # Prefer uploaded ckpt if provided; otherwise use text path
                    tmp_ckpt_path = None
                    if ckpt_upload is not None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tf:
                            tf.write(ckpt_upload.read())
                            tmp_ckpt_path = tf.name
                    ckpt_path = tmp_ckpt_path or ckpt_path_text.strip()

                    if not ckpt_path:
                        st.warning("Please provide or upload an Anomalib checkpoint (.ckpt). Using built-in mock instead.")
                        backend_effective = "Built-in (Mock)"
                    else:
                        backend_effective = "Anomalib"

                    if backend_effective == "Anomalib":
                        try:
                            # Lazy import to avoid hard dependency at app start
                            from anomalib.data import Folder as AnomFolder
                            from anomalib.engine import Engine as AnomEngine
                            mod = __import__("anomalib.models", fromlist=[model_name])
                            ModelCls = getattr(mod, model_name)

                            # Save image to a temp folder and use Folder datamodule
                            with tempfile.TemporaryDirectory() as td:
                                img_path = Path(td) / "sample.png"
                                Image.fromarray(image_array).save(img_path)

                                dm = AnomFolder(root=str(Path(td)), task="segmentation", image_size=int(image_size))
                                engine = AnomEngine()
                                model = ModelCls()

                                preds = engine.predict(datamodule=dm, model=model, ckpt_path=str(ckpt_path), return_predictions=True)
                                # Parse first prediction
                                p = preds[0] if isinstance(preds, (list, tuple)) and len(preds) > 0 else None
                                if p is None:
                                    raise RuntimeError("Anomalib produced no predictions.")

                                # Robust extraction of fields
                                img = p.get("image") if isinstance(p, dict) else getattr(p, "image")
                                score = float(p.get("pred_scores", p.get("pred_score", 0.0))) if isinstance(p, dict) else float(getattr(p, "pred_score", 0.0))
                                mask = p.get("pred_masks") if isinstance(p, dict) else getattr(p, "pred_mask", None)
                                if mask is None:
                                    raise RuntimeError("Anomalib did not return a segmentation mask. Try a segmentation-capable model like Patchcore or STFPM.")

                                # Normalize mask and create visualizations
                                m = mask.astype(np.float32)
                                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                                if smooth and cv2 is not None and int(smooth_kernel) > 1:
                                    k = int(smooth_kernel) if int(smooth_kernel) % 2 == 1 else int(smooth_kernel) + 1
                                    m = cv2.GaussianBlur(m, (k, k), 0)

                                # Heatmap
                                heat = (m * 255).astype(np.uint8)
                                cmap = {
                                    "JET": getattr(cv2, "COLORMAP_JET", 2),
                                    "TURBO": getattr(cv2, "COLORMAP_TURBO", 20),
                                    "HOT": getattr(cv2, "COLORMAP_HOT", 11),
                                    "PARULA": getattr(cv2, "COLORMAP_PARULA", getattr(cv2, "COLORMAP_TURBO", 20)),
                                }.get(colormap, getattr(cv2, "COLORMAP_JET", 2))
                                heat_color = cv2.applyColorMap(heat, cmap)
                                heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
                                overlay = cv2.addWeighted(heat_rgb, float(alpha), image_array, 1.0 - float(alpha), 0)

                                # Threshold
                                thr = float(np.quantile(m, float(dynamic_pct) / 100.0)) if dynamic else float(threshold)
                                bin_mask = (m >= thr).astype(np.uint8) * 255

                                # Contours and boxes
                                boxes, rboxes = [], []
                                cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for c in cnts:
                                    if cv2.contourArea(c) < float(min_region_area):
                                        continue
                                    x, y, w, h = cv2.boundingRect(c)
                                    boxes.append((int(x), int(y), int(w), int(h)))
                                    if draw_rotated:
                                        rect = cv2.minAreaRect(c)
                                        pts = cv2.boxPoints(rect)
                                        rboxes.append([(int(px), int(py)) for px, py in pts])

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
                                        "ckpt_path": str(ckpt_path),
                                        "threshold": float(thr),
                                        "dynamic": bool(dynamic),
                                        "dynamic_pct": float(dynamic_pct),
                                        "alpha": float(alpha),
                                        "colormap": colormap,
                                        "min_region_area": int(min_region_area),
                                        "smooth": bool(smooth),
                                        "smooth_kernel": int(smooth_kernel),
                                        "rotated": bool(draw_rotated),
                                    },
                                }
                        except ImportError as ie:
                            st.error("Anomalib is not installed. Run: pip install anomalib")
                            result = {}
                        except Exception as ae:
                            st.error("An error occurred running Anomalib. Falling back to built-in detector.")
                            st.exception(ae)
                            backend_effective = "Built-in (Mock)"
                    if backend_effective != "Anomalib":
                        # Fall back to built-in detector
                        detect_kwargs = {
                            "threshold": threshold,
                            "min_region_area": int(min_region_area),
                            "alpha": alpha,
                            "colormap": colormap,
                            "dynamic": dynamic,
                            "dynamic_pct": float(dynamic_pct),
                            "smooth": bool(smooth and cv2 is not None),
                            "smooth_kernel": int(smooth_kernel),
                            "rotated": bool(draw_rotated and cv2 is not None),
                        }
                        try:
                            sig = inspect.signature(detector.detect)
                            allowed = set(sig.parameters.keys())
                            filtered = {k: v for k, v in detect_kwargs.items() if k in allowed}
                        except Exception:
                            filtered = detect_kwargs
                        result = detector.detect(image_array, **filtered) if cv2 is not None else {}
                else:
                    # Built-in detector path
                    detect_kwargs = {
                        "threshold": threshold,
                        "min_region_area": int(min_region_area),
                        "alpha": alpha,
                        "colormap": colormap,
                        "dynamic": dynamic,
                        "dynamic_pct": float(dynamic_pct),
                        "smooth": bool(smooth and cv2 is not None),
                        "smooth_kernel": int(smooth_kernel),
                        "rotated": bool(draw_rotated and cv2 is not None),
                    }
                    try:
                        sig = inspect.signature(detector.detect)
                        allowed = set(sig.parameters.keys())
                        filtered = {k: v for k, v in detect_kwargs.items() if k in allowed}
                    except Exception:
                        filtered = detect_kwargs
                    result = detector.detect(image_array, **filtered) if cv2 is not None else {}

                # Prediction status and metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if result.get("is_anomaly", False):
                        st.error("🚨 Anomaly Detected")
                    else:
                        st.success("✅ No Anomaly Detected")
                with col2:
                    st.metric("Anomaly score", f"{result.get('anomaly_score', 0):.3f}")
                with col3:
                    st.metric("Confidence (max error)", f"{result.get('confidence', 0):.3f}")

                # Visual artifacts
                colA, colB, colC = st.columns(3)
                with colA:
                    st.image(image_array, caption="Original", width='stretch')
                    st.image(result["overlay"], caption="Overlay", width='stretch')
                with colB:
                    st.image(result["heatmap"], caption=f"Heatmap ({colormap})", width='stretch')
                with colC:
                    mask_caption = f"Mask (threshold={threshold:.2f})" if not dynamic else f"Mask (p={dynamic_pct:.1f}%)"
                    st.image(result["mask"], caption=mask_caption, width='stretch')

                # Bounding boxes preview
                boxed = image_array.copy()
                if cv2 is not None and result:
                    for x, y, w, h in result.get("boxes", []):
                        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    for pts in result.get("rotated_boxes", []):
                        pts_np = np.array(pts, dtype=np.int32)
                        cv2.polylines(boxed, [pts_np], isClosed=True, color=(0,255,0), thickness=2)
                st.image(boxed, caption="Bounding boxes" if result else "Bounding boxes (unavailable)", width='stretch')

                with st.expander("Advanced outputs"):
                    st.write("Parameters used:", result.get("params", {}))
                    st.code(str(result.get("params", {})), language="json")
                    # Optionally show raw error map statistics
                    em = result.get("error_map")
                    if em is not None:
                        st.write({"min": float(em.min()), "max": float(em.max()), "mean": float(em.mean())})
                    # Provide downloads defensively
                    enc_ok_mask, enc_mask = cv2.imencode('.png', result.get("mask"))
                    if enc_ok_mask:
                        st.download_button(
                            label="Download mask (PNG)",
                            data=enc_mask.tobytes(),
                            file_name="mask.png",
                            mime="image/png",
                        )
                    heatmap_bgr = cv2.cvtColor(result.get("heatmap"), cv2.COLOR_RGB2BGR)
                    enc_ok_heat, enc_heat = cv2.imencode('.png', heatmap_bgr)
                    if enc_ok_heat:
                        st.download_button(
                            label="Download heatmap (PNG)",
                            data=enc_heat.tobytes(),
                            file_name="heatmap.png",
                            mime="image/png",
                        )
            except Exception as e:
                import traceback
                st.error("An error occurred while analyzing the image.")
                st.exception(e)
                st.code("\n".join(traceback.format_exc().splitlines()[-15:]))

# Real-time video processing
st.subheader("Real-time Video Processing")
video_file = st.file_uploader("Upload a video...", type=["mp4", "avi"])

# Results gallery section (optional images from repo)
from pathlib import Path
with st.expander("Results gallery (ReverseDistillation / one_up / v0)"):
    try:
        assets_dir = Path(__file__).resolve().parents[2] / "assets" / "results" / "ReverseDistillation" / "one_up" / "v0"
        files = {
            "example_inference.png": "Example inference",
            "plot_right_classification.png": "Right classification",
            "plot_bad_classification.png": "Bad classification",
            "one_up_confusion_matrix.png": "Confusion matrix",
        }
        if assets_dir.exists():
            cols = st.columns(2)
            idx = 0
            for fname, title in files.items():
                fpath = assets_dir / fname
                if fpath.exists():
                    with cols[idx % 2]:
                        st.image(str(fpath), caption=title, use_container_width=True)
                    idx += 1
            if idx == 0:
                st.info(f"No images found in {assets_dir}")
        else:
            st.info(f"Place your result images under: {assets_dir}")
    except Exception:
        pass

if video_file is not None:
    st.video(video_file)
    # Here you would implement the video processing logic using the detector
    # This is a placeholder for the actual implementation
    st.write("Video processing feature is under development.")