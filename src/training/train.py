import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data.preprocessing import load_data
from src.models.cnn_model import create_cnn_model
from src.utils.config import get_training_config

def train_model():
    # Load configuration
    config = get_training_config()
    
    # Load and preprocess data
    images, labels = load_data(config['data_path'])
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=config['validation_split'], random_state=42)

    # Create the CNN model
    model = create_cnn_model(input_shape=X_train.shape[1:])
    
    # Compile the model
    model.compile(optimizer=config['optimizer'], loss=config['loss_function'], metrics=config['metrics'])
    
    # Train the model
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              epochs=config['epochs'], 
              batch_size=config['batch_size'])
    
    # Save the trained model
    model.save(os.path.join(config['model_save_path'], 'anomaly_detection_model.h5'))

if __name__ == "__main__":
    train_model()