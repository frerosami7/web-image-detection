def plot_training_history(history):
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def visualize_predictions(images, predictions, titles=None):
    import matplotlib.pyplot as plt

    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i] if titles else f'Prediction: {predictions[i]}')
        plt.axis('off')
    plt.show()


def display_image(image, title=None):
    import matplotlib.pyplot as plt

    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()