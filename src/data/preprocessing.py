def resize_image(image, target_size):
    from PIL import Image
    return image.resize(target_size, Image.ANTIALIAS)

def normalize_image(image):
    import numpy as np
    return np.array(image) / 255.0

def split_dataset(images, labels, train_size=0.8):
    from sklearn.model_selection import train_test_split
    return train_test_split(images, labels, train_size=train_size, random_state=42)