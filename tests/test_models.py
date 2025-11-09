import unittest
from src.models.cnn_model import CNNModel
from src.models.autoencoder import Autoencoder

class TestModels(unittest.TestCase):

    def setUp(self):
        self.cnn_model = CNNModel(input_shape=(128, 128, 3), num_classes=2)
        self.autoencoder = Autoencoder(input_shape=(128, 128, 3))

    def test_cnn_model_build(self):
        model = self.cnn_model.build()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 5)  # Example: Adjust based on actual architecture

    def test_autoencoder_model_build(self):
        model = self.autoencoder.build()
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 4)  # Example: Adjust based on actual architecture

    def test_cnn_model_compile(self):
        model = self.cnn_model.build()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.assertIn('accuracy', model.metrics_names)

    def test_autoencoder_model_compile(self):
        model = self.autoencoder.build()
        model.compile(optimizer='adam', loss='mse')
        self.assertIn('loss', model.metrics_names)

if __name__ == '__main__':
    unittest.main()