from gradio import Interface, inputs, outputs
import numpy as np
import cv2
from src.inference.detector import load_model, detect_anomalies

# Load the pre-trained model
model = load_model('path/to/your/model')

def predict(image):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model.input_shape[1], model.input_shape[2]))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize the image

    # Detect anomalies
    anomalies = detect_anomalies(model, image)
    return anomalies

# Define the Gradio interface
iface = Interface(
    fn=predict,
    inputs=inputs.Image(type="numpy"),
    outputs=outputs.Label(num_top_classes=2),
    title="Image Anomaly Detection",
    description="Upload an image to detect anomalies."
)

if __name__ == "__main__":
    iface.launch()