from neuro_disease_detector.neuro_training.segmentation_consesus import consensus
import unittest
import os
from pathlib import Path
import numpy as np

# Determine the absolute path of the current file
folder_path = os.path.dirname(os.path.abspath(__file__))

# Define a test case for segmentation-consensus
class TestSegmentationConsensus(unittest.TestCase):

    def setUp(self):
        """
        Set up the required variables for testing.
        - `self.yolo_model`: Path to the YOLO model to be tested.
        - `self.file_path`: Path to the NIfTI file to be tested.
        """

        self.file_path = Path(folder_path) / "MSLesSeg-Dataset" / "train" / "P1" / "T1" / "P1_T1_FLAIR.nii"
        self.yolo_model = Path(folder_path) / "yolo_test.pt"

    def test_consensus(self):
        """
        Test the performance of the `consensus` function.
        - Compare the model's output against the expected output.
        - Validate the number of votes for each class.
        """

        # Call the function under test and obtain the actual output
        votes = consensus(self.file_path, self.yolo_model)

        # Count the number of votes for each class
        count_0 = np.count_nonzero(votes == 0)
        count_1 = np.count_nonzero(votes == 1)
        count_2 = np.count_nonzero(votes == 2)
        count_3 = np.count_nonzero(votes == 3)

        # Assert that the number of votes for each class matches the
        self.assertEqual(count_0, 7134645)
        self.assertEqual(count_1, 58823)
        self.assertEqual(count_2, 14836)
        self.assertEqual(count_3, 12728)
        self.assertEqual(count_0 + count_1 + count_2 + count_3, votes.size)
