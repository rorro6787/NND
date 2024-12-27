from neuro_disease_detector.neuro_training.cross_validation import test_neuro_system
import unittest
import os
from pathlib import Path
import numpy as np

# Determine the absolute path of the current file
folder_path = os.path.dirname(os.path.abspath(__file__))

# Define a test case for cross-validation
class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        """
        Set up the required variables for testing.
        - `self.dataset_path`: The directory path where the dataset is located.
        - `self.fold`: Specifies the fold to be tested.
        - `self.yolo_model`: Path to the YOLO model to be tested.
        """

        self.dataset_path = folder_path
        self.fold = "fold1"
        self.yolo_model = Path(folder_path) / "yolo_test.pt"

    def test_model_validation(self):
        """
        Test the performance of the `test_neuro_system` function.
        - Compare the model's output against the expected output.
        - Validate key metrics such as TP, FP, TN, FN, Recall, Precision, Accuracy, Sensibility, IOU, and F1 Score.
        """

        # Define the expected output from the model validation
        expected_output = {
            'TP': np.int64(962),                
            'FP': np.int64(2439),              
            'TN': np.int64(471826),           
            'FN': np.int64(885),              
            'Recall': np.float64(0.5208446128857607),
            'Precision': np.float64(0.2828579829461923), 
            'Acc': np.float64(0.9930184494404678),     
            'Sensibility': np.float64(0.22445170321978536),
            'IOU': np.float64(0.22445170321978536),   
            'F1 Score': np.float64(0.36661585365853655) 
        }

        # Call the function under test and obtain the actual output
        model_output = test_neuro_system(self.dataset_path, self.fold, self.yolo_model)

        # Assert that the model's output matches the expected output
        self.assertDictEqual(model_output, expected_output)
