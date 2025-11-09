# Configuration settings for the Image Anomaly Detection project

import os

class Config:
    # General settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

    # Model settings
    MODEL_NAME = 'cnn_model'
    INPUT_SHAPE = (128, 128, 3)  # Example input shape for CNN
    NUM_CLASSES = 2  # Adjust based on the number of anomaly types

    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # Paths for datasets
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train')
    VALIDATION_DATA_PATH = os.path.join(DATA_DIR, 'validation')
    TEST_DATA_PATH = os.path.join(DATA_DIR, 'test')

    # Logging settings
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Web interface settings
    GRADIO_INTERFACE = True  # Set to True to enable Gradio interface
    STREAMLIT_INTERFACE = False  # Set to True to enable Streamlit interface

    # API settings
    API_VERSION = 'v1'
    API_PORT = 5000

    # Real-time video processing settings
    VIDEO_SOURCE = 0  # Default to webcam; change to video file path if needed
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Additional settings can be added as needed

def load_config():
    """Load and return configuration settings"""
    return {
        'BASE_DIR': Config.BASE_DIR,
        'DATA_DIR': Config.DATA_DIR,
        'MODEL_DIR': Config.MODEL_DIR,
        'OUTPUT_DIR': Config.OUTPUT_DIR,
        'MODEL_NAME': Config.MODEL_NAME,
        'INPUT_SHAPE': Config.INPUT_SHAPE,
        'NUM_CLASSES': Config.NUM_CLASSES,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'EPOCHS': Config.EPOCHS,
        'LEARNING_RATE': Config.LEARNING_RATE,
        'TRAIN_DATA_PATH': Config.TRAIN_DATA_PATH,
        'VALIDATION_DATA_PATH': Config.VALIDATION_DATA_PATH,
        'TEST_DATA_PATH': Config.TEST_DATA_PATH,
        'LOG_DIR': Config.LOG_DIR,
        'LOG_LEVEL': Config.LOG_LEVEL,
        'GRADIO_INTERFACE': Config.GRADIO_INTERFACE,
        'STREAMLIT_INTERFACE': Config.STREAMLIT_INTERFACE,
        'API_VERSION': Config.API_VERSION,
        'API_PORT': Config.API_PORT,
        'PORT': Config.API_PORT,
        'VIDEO_SOURCE': Config.VIDEO_SOURCE,
        'FRAME_WIDTH': Config.FRAME_WIDTH,
        'FRAME_HEIGHT': Config.FRAME_HEIGHT
    }