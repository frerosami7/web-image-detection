import streamlit as st
import numpy as np
import sys
from pathlib import Path
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
    from anomalib.deploy import TorchInferencer
except Exception:
    TorchInferencer = None

# Anomalib-only backend
try:
    from src.inference.anomalib_detector import single_image_predict
except ModuleNotFoundError:
    from inference.anomalib_detector import single_image_predict  # type: ignore

st.set_page_config(page_title="Image Anomaly Detection", layout="wide")

st.title("Image Anomaly Detection")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    # Convert PIL to numpy RGB uint8
    image_array = np.array(image.convert("RGB"))

    # Sidebar parameters
    with st.sidebar:
        st.header("Anomalib Torch Model")
        image_size = st.number_input("Image size (resize before infer)", min_value=64, max_value=1024, value=256, step=32)
        torch_model_text = st.text_input("Torch model path (.pt)", value="", help="Absolute or repo-relative path on server")
        torch_model_upload = st.file_uploader("Or upload Torch model", type=["pt"], accept_multiple_files=False)
        # Auto-discover .pt files under a directory
        st.subheader("Discover models")
        models_dir = st.text_input("Scan directory", value="checkpoints", help="Search recursively for .pt models")
        discovered_paths = []
        try:
            mdir = Path(models_dir)
            if mdir.exists():
                discovered_paths = [str(p) for p in mdir.rglob("*.pt")]
        except Exception:
            discovered_paths = []
        selected_model = st.selectbox("Discovered .pt models", options=[""] + discovered_paths, index=0)
        if st.button("Clear cached models"):
            st.session_state["inferencer_cache"] = {}

        # Optional: download model from URL to bypass upload limits
        st.subheader("Download model from URL")
        dl_url = st.text_input("Model URL (.pt)", value="", help="Direct link to a .pt file")
        dl_name = st.text_input("Save as (optional)", value="", help="Filename under checkpoints/ (leave empty to auto-use URL name)")
        if st.button("Download .pt to checkpoints") and dl_url.strip():
            try:
                import shutil
                from urllib.request import urlopen
                url = dl_url.strip()
                fname = dl_name.strip() or Path(url).name or "downloaded_model.pt"
                dest = Path(models_dir or "checkpoints") / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(url) as resp, open(dest, "wb") as out:
                    shutil.copyfileobj(resp, out)
                st.success(f"Downloaded to {dest}")
            except Exception as e:
                st.error(f"Download failed: {e}")

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
        draw_rotated = st.checkbox("Rotated boxes (cv2 only)", value=False, help="Use minimum-area rotated rectangles")
        colormap = st.selectbox("Colormap", ["JET", "TURBO", "HOT", "PARULA"], index=0)

    # Detect anomalies
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner('Analyzing image...'):
            try:
                # Use uploaded torch model if provided; otherwise text path
                tmp_model_path = None
                if torch_model_upload is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tf:
                        tf.write(torch_model_upload.read())
                        tmp_model_path = tf.name
                # Priority: uploaded > discovered dropdown > manual text
                torch_model_path = tmp_model_path or (selected_model or torch_model_text.strip())

                if not torch_model_path:
                    st.warning("Please provide or upload an Anomalib Torch model (.pt).")
                    result = {}
                else:
                    # Cache/reuse inferencer
                    infer_cache = st.session_state.setdefault("inferencer_cache", {})
                    infer_key = f"{torch_model_path}"
                    cached_infer = infer_cache.get(infer_key)
                    if cached_infer is None:
                        if TorchInferencer is None:
                            st.error("Anomalib TorchInferencer not available. pip install anomalib")
                            result = {}
                        else:
                            try:
                                cached_infer = TorchInferencer(path=torch_model_path, device="auto")
                                infer_cache[infer_key] = cached_infer
                            except Exception as e:
                                st.error(f"Failed to load model: {e}")
                                cached_infer = None

                    if TorchInferencer is None or cached_infer is None:
                        # Abort prediction gracefully if anomalib unavailable or model failed to load
                        result = {}
                    else:
                        result = single_image_predict(
                            image_rgb=image_array,
                            model_name="",
                            ckpt_path="",
                            image_size=int(image_size),
                            threshold_mode=("dynamic" if dynamic else "static"),
                            dynamic_pct=float(dynamic_pct),
                            static_threshold=float(threshold),
                            min_region_area=int(min_region_area),
                            alpha=float(alpha),
                            colormap=colormap,
                            rotated=bool(draw_rotated),
                            smooth=bool(smooth),
                            smooth_kernel=int(smooth_kernel),
                            torch_model_path=torch_model_path,
                            inferencer=cached_infer,
                        )

                # Guard: if some artifacts are missing, warn and render only what's available
                required_keys = ["heatmap", "mask", "overlay"]
                missing = [k for k in required_keys if k not in result]
                incomplete = False
                if missing:
                    incomplete = True
                    st.warning(f"Prediction incomplete. Missing keys: {missing}. Check checkpoint/model compatibility.")

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
                    overlay_img = result.get("overlay")
                    if overlay_img is not None:
                        st.image(overlay_img, caption="Overlay", width='stretch')
                    else:
                        st.info("Overlay unavailable")
                with colB:
                    heatmap_img = result.get("heatmap")
                    if heatmap_img is not None:
                        st.image(heatmap_img, caption=f"Heatmap ({colormap})", width='stretch')
                    else:
                        st.info("Heatmap unavailable")
                with colC:
                    mask_img = result.get("mask")
                    if mask_img is not None:
                        mask_caption = f"Mask (threshold={threshold:.2f})" if not dynamic else f"Mask (p={dynamic_pct:.1f}%)"
                        st.image(mask_img, caption=mask_caption, width='stretch')
                    else:
                        st.info("Mask unavailable")

                # Bounding boxes preview
                boxed = image_array.copy()
                if result:
                    # Draw boxes with cv2 if available, otherwise numpy outline
                    if cv2 is not None:
                        for x, y, w, h in result.get("boxes", []):
                            cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        for pts in result.get("rotated_boxes", []):
                            pts_np = np.array(pts, dtype=np.int32)
                            cv2.polylines(boxed, [pts_np], isClosed=True, color=(0,255,0), thickness=2)
                    else:
                        for x, y, w, h in result.get("boxes", []):
                            boxed[y:y+h, x:x+1] = [255,0,0]
                            boxed[y:y+h, x+w-1:x+w] = [255,0,0]
                            boxed[y:y+1, x:x+w] = [255,0,0]
                            boxed[y+h-1:y+h, x:x+w] = [255,0,0]
                st.image(boxed, caption="Bounding boxes" if result else "Bounding boxes (unavailable)", width='stretch')

                with st.expander("Advanced outputs"):
                    st.write("Parameters used:", result.get("params", {}))
                    st.code(str(result.get("params", {})), language="json")
                    # Optionally show raw error map statistics
                    em = result.get("error_map")
                    if em is not None:
                        st.write({"min": float(em.min()), "max": float(em.max()), "mean": float(em.mean())})
                    # Provide downloads (skip if cv2 missing)
                    if cv2 is not None:
                        if result.get("mask") is not None:
                            enc_ok_mask, enc_mask = cv2.imencode('.png', result.get("mask"))
                            if enc_ok_mask:
                                st.download_button(
                                    label="Download mask (PNG)",
                                    data=enc_mask.tobytes(),
                                    file_name="mask.png",
                                    mime="image/png",
                                )
                        if result.get("heatmap") is not None:
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