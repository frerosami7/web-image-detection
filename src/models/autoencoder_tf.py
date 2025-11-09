from tensorflow.keras import layers, models

class Autoencoder:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        # Encoder
        input_img = layers.Input(shape=self.input_shape)
        x = layers.Flatten()(input_img)
        x = layers.Dense(128, activation='relu')(x)
        encoded = layers.Dense(64, activation='relu')(x)

        # Decoder
        x = layers.Dense(128, activation='relu')(encoded)
        x = layers.Dense(self.input_shape[0] * self.input_shape[1], activation='sigmoid')(x)
        decoded = layers.Reshape(self.input_shape)(x)

        # Autoencoder model
        autoencoder = models.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder

    def train(self, x_train, x_val, epochs=50, batch_size=256):
        self.model.fit(x_train, x_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(x_val, x_val))

    def encode(self, x):
        encoder = models.Model(self.model.input, self.model.layers[2].output)
        return encoder.predict(x)

    def decode(self, encoded_imgs):
        decoder_input = layers.Input(shape=(64,))
        x = self.model.layers[3](decoder_input)
        x = self.model.layers[4](x)
        decoder = models.Model(decoder_input, x)
        return decoder.predict(encoded_imgs)