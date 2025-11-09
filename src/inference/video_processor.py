import cv2
import numpy as np
from src.models.cnn_model import load_model  # Assuming a function to load the trained model
from src.inference.detector import detect_anomalies  # Assuming a function to detect anomalies

class VideoProcessor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame for the model
            processed_frame = self.preprocess_frame(frame)

            # Detect anomalies
            anomalies = detect_anomalies(processed_frame, self.model)

            # Display results
            self.display_results(frame, anomalies)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        # Resize and normalize the frame as required by the model
        frame_resized = cv2.resize(frame, (224, 224))  # Example size
        frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
        return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

    def display_results(self, frame, anomalies):
        # Draw bounding boxes or annotations on the frame
        for anomaly in anomalies:
            x, y, w, h = anomaly  # Assuming anomaly is defined by bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)