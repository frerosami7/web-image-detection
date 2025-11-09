from typing import Tuple, List, Any


def resize_image(image, target_size: Tuple[int, int]):
    from PIL import Image
    # Use modern resampling to avoid deprecation warnings
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    return image.resize(target_size, resample)


def normalize_image(image):
    import numpy as np
    return (np.array(image).astype('float32')) / 255.0


def preprocess_image(path: str, target_size: Tuple[int, int] = (128, 128)):
    """Load, resize, and normalize an image from a filesystem path.

    Returns a numpy array in [0,1].
    """
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = resize_image(img, target_size)
    return normalize_image(img)


def split_dataset(*args, **kwargs):
    """Flexible dataset split helper.

    Supports two signatures:
    1) images, labels, train_size=0.8 -> returns X_train, X_val, y_train, y_val
    2) dataset, val_size=0.2 -> returns train_set, val_set
    """
    from sklearn.model_selection import train_test_split

    # Case 1: (images, labels)
    if len(args) == 2 and not isinstance(args[1], (float, int)):
        images, labels = args
        train_size = kwargs.get('train_size', 0.8)
        return train_test_split(images, labels, train_size=train_size, random_state=42, shuffle=True)

    # Case 2: (dataset,)
    if len(args) == 1:
        dataset = args[0]
        val_size = kwargs.get('val_size', 0.2)
        train_size = 1.0 - float(val_size)
        train_set, val_set = train_test_split(dataset, train_size=train_size, random_state=42, shuffle=True)
        return train_set, val_set

    raise ValueError("split_dataset expects (images, labels, train_size=...) or (dataset, val_size=...)")