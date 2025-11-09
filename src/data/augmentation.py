from keras.preprocessing.image import ImageDataGenerator

def create_data_augmentation_generator(rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True,
                                       fill_mode='nearest'):
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )
    return datagen

def augment_images(images):
    augmented_images = []
    for img in images:
        img = img.reshape((1,) + img.shape)  # Reshape for the generator
        for batch in create_data_augmentation_generator().flow(img, batch_size=1):
            augmented_images.append(batch[0])
            if len(augmented_images) >= 5:  # Generate 5 augmented images
                break
    return augmented_images