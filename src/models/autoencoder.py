import numpy as np


class _MockKerasModel:
    def __init__(self, num_layers: int):
        self.layers = [object() for _ in range(num_layers)]
        self.metrics_names = []

    def compile(self, optimizer=None, loss=None, metrics=None):
        self.metrics_names = []
        if loss is not None:
            self.metrics_names.append('loss')
        if metrics:
            self.metrics_names.extend(list(metrics))


class Autoencoder:
    """Mock Autoencoder for demonstration and tests (no TF dependency)."""
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        print("⚠️ Using Mock Autoencoder - TensorFlow not available")

    # --- Test-friendly API ---
    def build(self):
        self.model = _MockKerasModel(num_layers=4)
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=None):
        if self.model is None:
            self.build()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

    # --- Backward-compatible mock methods ---
    def build_model(self):
        print("Mock autoencoder built successfully")
        return self

    def predict(self, image):
        return np.random.random((1, *self.input_shape))

    def train(self, x_train, x_val, epochs=50, batch_size=256):
        print(f"Mock training completed: {epochs} epochs")