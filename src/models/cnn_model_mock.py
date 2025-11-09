import numpy as np
from PIL import Image
import cv2

class CNNModel:
    """Mock CNN Model for demonstration purposes"""
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        print("⚠️ Using Mock CNN Model - TensorFlow not available")
    
    def predict(self, image):
        """Mock prediction - returns random anomaly detection results"""
        # Simulate anomaly detection
        anomaly_score = np.random.random()
        is_anomaly = anomaly_score > 0.5
        
        # Return prediction in expected format
        if is_anomaly:
            return np.array([[0.2, 0.8]])  # [normal_prob, anomaly_prob]
        else:
            return np.array([[0.8, 0.2]])  # [normal_prob, anomaly_prob]
    
    def build_model(self):
        """Mock model building"""
        print("Mock model built successfully")
        return self
    
    def compile_model(self, learning_rate=0.001):
        """Mock model compilation"""
        print("Mock model compiled")
        return self