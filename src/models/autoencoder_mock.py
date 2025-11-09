import numpy as np

class Autoencoder:
    """Mock Autoencoder for demonstration purposes"""
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        print("⚠️ Using Mock Autoencoder - TensorFlow not available")
    
    def build_model(self):
        """Mock model building"""
        print("Mock autoencoder built successfully")
        return self
    
    def predict(self, image):
        """Mock prediction"""
        return np.random.random((1, *self.input_shape))
    
    def train(self, x_train, x_val, epochs=50, batch_size=256):
        """Mock training"""
        print(f"Mock training completed: {epochs} epochs")