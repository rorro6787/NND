from neuro_disease_detector.logger import get_logger

import pandas as pd
import zipfile
import gdown
import os

logger = get_logger(__name__)

FOLD_TO_PATIENT = { "fold1": (1, 7), "fold2": (7, 14), "fold3": (14, 24), "fold4": (24, 41), "fold5": (41, 54) }
TIMEPOINTS_PATIENT = [3,4,4,3,2,3,2,2,3,2,2,2,4,4,1,1,1,1,4,3,1,1,2,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2]                      

def get_timepoints_patient(pd: int) -> int:
    """Returns the timepoints for a given patient."""
    return TIMEPOINTS_PATIENT[pd-1]

def get_patient_by_test_id(test_id: int | str) -> str:
    """ Given a test ID and a list with the number of tests per patient, return the patient to which the test belongs."""
    
    # Convert the test ID to an integer
    test_id = int(test_id)
    current_id = 0

    # Iterate over the number of tests per patient
    for i, num_tests in enumerate(TIMEPOINTS_PATIENT):
        current_id += num_tests
        if test_id <= current_id:
            return i + 1
    # If the test ID is not found, return -1
    return -1

def get_patients_split(split: str) -> tuple:
    """Returns the list of patients for a given split (e.g., train, test)."""
    return FOLD_TO_PATIENT[split]

def split_assign(pd: int) -> str:
    """Assign a patient to a fold based on the patient ID."""

    if pd <= 0 or pd >= 54:
        raise ValueError(f"Invalid patient ID: {pd}")
    
    # Define the boundaries for each fold
    folds = [FOLD_TO_PATIENT[f"fold{i}"][0] for i in range(1, 6)] 

    # Assign the patient to a fold based on their ID
    for i, start in enumerate(folds[:-1]):
        # If the patient ID is within the range of the current fold, return the fold
        if pd >= start and pd < folds[i + 1]:
            return f"fold{i + 1}"
    # If the patient ID is not within the range of any fold, return "test"
    return "fold5"

def download_dataset_from_cloud(folder_name: str, url: str, extract_folder: bool = True) -> None:
    """Downloads and extracts a dataset from a cloud storage URL."""

    logger.info(f"Downloading {folder_name} for yolo/nnUNet pipeline...")
    if os.path.exists(folder_name):
        return
    
    # Name of the ZIP file to save locally
    dataset_zip = f"{folder_name}.zip"

    # Download the dataset from the cloud storage URL
    gdown.download(url, dataset_zip, quiet=False)
    
    # Extract the dataset from the ZIP file
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        if extract_folder:
            zip_ref.extractall(folder_name)
        else:
            zip_ref.extractall()

    # Remove the ZIP file after extraction
    os.remove(dataset_zip)

def write_results_csv(csv_path: str, algorithm, instance, metric_name, execution_id, metric_value):
    """Write the results of an algorithm execution to a CSV file."""

    # Create a new row with the algorithm, instance, metric name, execution ID, and metric value
    new_row = pd.DataFrame([{
        "Algorithm": algorithm,
        "Instance": instance,
        "MetricName": metric_name,
        "ExecutionId": int(execution_id), 
        "MetricValue": metric_value,
    }])

    # Create a new CSV file if it doesn't exist
    data = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=new_row.columns)

    # Append the new row to the existing data
    data = pd.concat([data.dropna(axis=1, how='all'), new_row], ignore_index=True)

    # Save the updated data to the CSV file
    data.to_csv(csv_path, index=False)