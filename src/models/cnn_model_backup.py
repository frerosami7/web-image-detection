from tensorflow.keras import layers, models

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def compile_model(model, learning_rate=0.001):
    model.compile(optimizer=models.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, train_data, train_labels, validation_data, validation_labels, epochs=10, batch_size=32):
    history = model.fit(train_data, train_labels, 
                        validation_data=(validation_data, validation_labels), 
                        epochs=epochs, 
                        batch_size=batch_size)
    return history