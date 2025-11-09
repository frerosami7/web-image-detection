import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class CNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.build_model()
    
    def build_model(self):
        """Build the CNN model architecture"""
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.compile_model()
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def predict(self, image):
        """Predict anomalies in the given image"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Ensure image is properly shaped
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image)
        return predictions
    
    def train(self, train_data, train_labels, validation_data=None, validation_labels=None, epochs=10, batch_size=32):
        """Train the model"""
        if validation_data is not None:
            validation_data = (validation_data, validation_labels)
        
        history = self.model.fit(
            train_data, train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)

# Legacy functions for backward compatibility
def build_cnn_model(input_shape, num_classes):
    """Legacy function - use CNNModel class instead"""
    cnn = CNNModel(input_shape, num_classes)
    return cnn.model

def compile_model(model, learning_rate=0.001):
    """Legacy function - use CNNModel class instead"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

def train_model(model, train_data, train_labels, validation_data, validation_labels, epochs=10, batch_size=32):
    """Legacy function - use CNNModel class instead"""
    history = model.fit(
        train_data, train_labels, 
        validation_data=(validation_data, validation_labels), 
        epochs=epochs, 
        batch_size=batch_size
    )
    return history