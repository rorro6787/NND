import os
from pathlib import Path
import shutil
import gdown
import zipfile

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def prepare_dataset(dataset_path: str) -> None:
    """
    Processes a dataset of medical images stored in a hierarchical directory structure.

    This function navigates through the training dataset located in 'MSLesSeg-Dataset/train',
    identifying patients and their respective timepoints. It searches for files with a
    '.gz' extension, decompresses them if necessary, and collects the original compressed
    files for deletion if their decompressed counterparts already exist.

    The directory structure expected is as follows:
    - MSLesSeg-Dataset/
        - train/
            - Patient1/
                - Timepoint1/
                    - file1.nii.gz
                    - file2.nii.gz
                - Timepoint2/
                    - file3.nii.gz
            - Patient2/
                - Timepoint1/
                    - file4.nii.gz
                    - file5.nii.gz

    The function does not return any value. Instead, it modifies the file system by deleting files.

    Raises:
        OSError: If there are issues with file or directory access during the execution.
    """
    
    logging.info("Processing the dataset...")

    # Define the dataset path with the 'train' subdirectory
    dataset_path = Path(dataset_path) / "train"
    files_to_remove = set()

    # Loop through each patient directory inside the train folder
    for patient_directory in dataset_path.iterdir():
        if not patient_directory.is_dir():
            continue
        
        # Remove corrupted patient directory if found
        if patient_directory.name == "P30":
            logging.warning(f"Removing corrupted patient directory: {patient_directory}")
            shutil.rmtree(patient_directory)
            continue
        
        # Loop through each timepoint directory within the patient directory
        for timepoint_directory in patient_directory.iterdir():
            if not timepoint_directory.is_dir():
                continue
            
            # List all files in the current timepoint directory
            files_in_timepoint = os.listdir(timepoint_directory)

            for file_name in files_in_timepoint:
                # Process only files with the '.gz' extension
                if not file_name.endswith(".gz"):
                    continue
                
                # Construct the expected decompressed file name and path
                decompressed_file_name = file_name[:-3]
                decompressed_file_path = timepoint_directory / decompressed_file_name
                
                # Check if the decompressed file already exists
                if decompressed_file_path.is_file():
                    # Schedule the .gz file for deletion
                    files_to_remove.add(timepoint_directory / file_name)
                else:
                    # Decompress the .gz file
                    os.system(f"gunzip {timepoint_directory / file_name}")

    # Remove redundant .gz files if any were found
    if files_to_remove:
        logging.info("Removing redundant .gz files:")
        for file_path in files_to_remove:
            logging.info(f"Deleting {file_path}")
            os.remove(file_path)
    else:
        logging.info("No redundant .gz files found to delete.")

    logging.info("Dataset processing finished.")

def download_dataset_from_cloud(url: str, folder_name: str) -> None:
    """
    Downloads and extracts a dataset from a cloud storage URL.
    
    Parameters:
    - url (str): The URL of the file to download.
    - folder_name (str): The folder where the dataset will be extracted.
    
    Raises:
    - FileNotFoundError: If the downloaded file cannot be found.
    - zipfile.BadZipFile: If the ZIP file is invalid or corrupted.
    """

    # Name of the ZIP file to save locally
    dataset_zip = f"{folder_name}.zip"

    # Download the ZIP file
    logging.info("Starting dataset download...")
    gdown.download(url, dataset_zip, quiet=False)
    logging.info(f"File downloaded as {dataset_zip}")
    
    # Extract the contents
    logging.info("Extracting dataset...")
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall()
    logging.info("Dataset extracted successfully.")

    # Delete the ZIP file
    os.remove(dataset_zip)
    logging.info("Temporary ZIP file removed.")

def patients_timepoints() -> dict:
    """
    Counts the number of timepoints for each patient in the dataset.

    This function traverses through the training directory within the given dataset path
    and counts how many timepoints (subdirectories) exist for each patient (subdirectory).
    It skips non-directory files.

    Returns:
    --------
    dict
        A dictionary where keys are patient IDs (directory names) and values are the
        number of timepoints (subdirectories) for each patient.

    Example:
    --------
    >>> patients_timepoints('/path/to/dataset')
    {'patient_01': 3, 'patient_02': 4}
    """
    
    # Define the dataset path and training directory
    dataset_path = "MSLesSeg-Dataset"
    training_directory = Path(os.path.join(dataset_path, "train"))

    # Initialize an empty dictionary to store the count of timepoints per patient
    patients = {}

    # Iterate through each subdirectory in the training directory (each representing a patient)
    for patient_directory in training_directory.iterdir():
        # Skip files that are not directories (we are only interested in directories for patients)
        if not patient_directory.is_dir():
            continue

        # Initialize the count of timepoints for this patient to 0
        patients[patient_directory.name] = 0

        # Iterate through each subdirectory inside the patient's directory (representing timepoints)
        for timepoint_directory in patient_directory.iterdir():

            # Skip files that are not directories (only interested in subdirectories as timepoints)
            if not timepoint_directory.is_dir():
                continue

            # Increment the count of timepoints for this patient
            patients[patient_directory.name] += 1

    # Log the result showing the number of timepoints per patient
    logging.info(f"Timepoints per patient: {patients}")
    return patients

def extract_dataset() -> bool:
    """
    Downloads and prepares the MSLesSeg-Dataset if not already extracted.
    
    Returns:
        bool: True if the dataset is ready, False if an error occurred.
    """

    url = "https://drive.google.com/uc?id=1i3JgXqRF43WNLlScDOPaolDPJpxNMYxc"
    folder_name = "MSLesSeg-Dataset"

    # Check if the folder already exists
    if os.path.exists(folder_name):
        logging.info(f"The dataset folder '{folder_name}' already exists.")
        patients_timepoints()
        return True
    
    try:
        # Download and prepare the dataset
        logging.info("Downloading dataset...")
        download_dataset_from_cloud(url, folder_name)
        logging.info("Preparing dataset...")
        prepare_dataset(folder_name)
        logging.info("Dataset downloaded and prepared successfully.")
        patients_timepoints()
        return True
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found error: {fnf_error}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return False

if __name__ == "__main__":
    extract_dataset()
