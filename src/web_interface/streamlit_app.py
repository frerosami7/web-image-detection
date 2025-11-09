import streamlit as st
import numpy as np
try:
    import cv2
except Exception as e:
    cv2 = None
    CV2_IMPORT_ERROR = str(e)
from PIL import Image

# Use mock-friendly autoencoder and artifact-capable detector
from src.models.autoencoder import Autoencoder
from src.inference.detector import Detector

st.set_page_config(page_title="Image Anomaly Detection", layout="wide")

# Initialize models
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
        st.header("Parameters")
        threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01, help="Threshold on normalized error map")
        min_region_area = st.number_input("Min region area (px)", min_value=0, value=50, step=5,
                                          help="Small regions are removed")
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.45, 0.01, help="Blend factor for heatmap overlay")
        colormap = st.selectbox("Colormap", ["JET", "TURBO", "HOT", "PARULA"], index=0)

    # Detect anomalies
    if cv2 is None:
        st.error("OpenCV failed to import on this platform. Details: " + CV2_IMPORT_ERROR)
    elif st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner('Analyzing image...'):
            try:
                # Analyze and produce visual artifacts
                result = detector.detect(
                    image_array,
                    threshold=threshold,
                    min_region_area=int(min_region_area),
                    alpha=alpha,
                    colormap=colormap,
                ) if cv2 is not None else {}

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
                    st.image(result["mask"], caption=f"Mask (threshold={threshold:.2f})", width='stretch')

                # Bounding boxes preview
                boxed = image_array.copy()
                if cv2 is not None and result:
                    for x, y, w, h in result.get("boxes", []):
                        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 2)
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

if video_file is not None:
    st.video(video_file)
    # Here you would implement the video processing logic using the detector
    # This is a placeholder for the actual implementation
    st.write("Video processing feature is under development.")