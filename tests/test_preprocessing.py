import unittest
from src.data.preprocessing import preprocess_image, split_dataset

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_image(self):
        # Test the image preprocessing function
        input_image = 'path/to/test/image.jpg'
        processed_image = preprocess_image(input_image)
        
        # Check if the processed image is not None
        self.assertIsNotNone(processed_image)
        # Add more assertions based on expected output properties

    def test_split_dataset(self):
        # Test the dataset splitting function
        dataset = ['image1.jpg', 'image2.jpg', 'image3.jpg']
        train_set, val_set = split_dataset(dataset, val_size=0.2)
        
        # Check if the lengths of the train and validation sets are correct
        self.assertEqual(len(train_set), 2)
        self.assertEqual(len(val_set), 1)
        # Add more assertions based on expected output properties

if __name__ == '__main__':
    unittest.main()