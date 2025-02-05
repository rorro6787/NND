import os
import numpy as np
import json
from sklearn.model_selection import KFold
from pathlib import Path

import shutil

num_timepoints_per_patient = [3,4,4,3,2,3,2,2,3,2,2,4,2,4,1,1,1,1,4,3,1,2,1,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2]

fold_to_patient = {
      "fold1": (1, 6),
      "fold2": (6, 12),
      "fold3": (12, 19),
      "fold4": (19, 28),
      "fold5": (28, 41),
      "test": (41, 54)
}

def split_assign(pd: int):
    if pd >= 1 and pd < 6:
        return "fold1"
    if pd >= 6 and pd < 12:
        return "fold2"
    if pd >= 12 and pd < 19:
        return "fold3"
    if pd >= 19 and pd < 28:
        return "fold4"
    if pd >= 28 and pd < 41:
        return "fold5"
    return "test"

def create_nnu_dataset(dataset_dir: str):
    # Define the path where the nnUNet dataset will be stored
    nnUNet_datapath = f"{os.getcwd()}/nnUNet_raw/Dataset024_MSLesSeg"
    dataset_path = f"{dataset_dir}/MSLesSeg-Dataset/train"

    # Create necessary directories for the nnUNet dataset
    os.makedirs(nnUNet_datapath, exist_ok=True)
    os.makedirs(f"{nnUNet_datapath}/imagesTr")  # Training images folder
    os.makedirs(f"{nnUNet_datapath}/imagesTs")  # Testing images folder
    os.makedirs(f"{nnUNet_datapath}/labelsTr")  # Training labels folder
    os.makedirs(f"{nnUNet_datapath}/labelsTs")  # Testing labels folder
    
    # Initialize a unique id counter for each subject
    id = 0

    # Iterate over the subjects in the dataset
    for pd in range(1, 54):
        # Skip the subject with id 30
        if pd == 30:
            continue

        # Define the path for a specific subject's folder
        pd_path = f"{dataset_path}/P{pd}"

        # Iterate over the 4 timepoints for each subject
        for td in range(1, 5):
            # Define the path for the timepoint folder
            td_path = f"{pd_path}/T{td}"

            # Break the loop if the timepoint folder doesn't exist
            if not os.path.exists(td_path):
                break

            # Increment the dataset ID for each subject/timepoint pair
            id += 1

            # Define the paths to the image and mask files
            flair_path = f"{td_path}/P{pd}_T{td}_FLAIR.nii"
            t1_path = f"{td_path}/P{pd}_T{td}_T1.nii"
            t2_path = f"{td_path}/P{pd}_T{td}_T2.nii"
            mask_path = f"{td_path}/P{pd}_T{td}_MASK.nii"

            # Assign the subject to either the 'train' or 'test' fold
            fold = split_assign(pd)  # Function to assign fold (train/test)
            train_test = "Ts" if fold == "test" else "Tr"  # Test or Train based on fold

            # Copy the images and mask to the appropriate directories
            shutil.copy(flair_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0000.nii.gz")
            shutil.copy(t1_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0001.nii.gz")
            shutil.copy(t2_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0002.nii.gz")
            shutil.copy(mask_path, f"{nnUNet_datapath}/labels{train_test}/BRATS_{id}.nii.gz")

def get_patient_by_test_id(test_id: int|str):
    """
    Given a test ID and a list with the number of tests per patient,
    return the patient to which the test belongs.

    Args:
        test_id (int | str): The ID of the test.
        tests_per_patient (list): A list where each element indicates the number of tests for each patient.

    Returns:
        str: The patient to which the test belongs.
    """

    # Ensure test_id is an integer (in case it's provided as a string)
    test_id = int(test_id)
    
    # Initialize a variable to track the cumulative number of tests encountered
    current_id = 0

    # Iterate through each patient and the number of tests they have
    for i, num_tests in enumerate(num_timepoints_per_patient):
        # Add the number of tests for the current patient to the cumulative count
        current_id += num_tests
        
        # If the test_id is within the range of tests for this patient, return their ID
        if test_id <= current_id:
            return f"P{i + 1}"

    # If no matching patient is found, return "ID not found"
    return "ID not found"





print(get_patient_by_test_id("30"))
dataset_path = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/nnu_net"
# create_nnu_dataset(dataset_path)