import gdown
import zipfile
import os

fold_to_patient = { "fold1": (1, 7), "fold2": (7, 14), "fold3": (14, 23), "fold4": (23, 37), "fold5": (37, 50), "test": (50, 54) }
timepoints_patient = [3,4,4,3,2,3,2,2,3,2,2,4,2,4,1,1,1,1,4,3,1,1,2,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2]
cwd = os.getcwd()

def get_timepoints_patient(pd: int):
    """Returns the timepoints for a given patient, adjusted by -1."""
    return timepoints_patient(pd-1)

def get_patients_split(split: str):
    """Returns the list of patients for a given split (e.g., train, test)."""
    return fold_to_patient[split]

def split_assign(pd: int):
        """
        Assign a patient to a fold based on the patient ID.

        Args:
            pd (int): The patient ID.

        Returns:
            str: The fold to which the patient belongs.

        Example:
            >>> from neuro_disease_detector.utils.utils_dataset import split_assign
            >>>
            >>> # Assign the patient to a fold based on their ID
            >>> fold = split_assign(1)
            >>> print(fold)
            fold1
        """

        # Define the boundaries for each fold
        folds = [1, 7, 14, 23, 37, 50]

        # Assign the patient to a fold based on their ID
        for i, start in enumerate(folds[:-1]):
            # If the patient ID is within the range of the current fold, return the fold
            if pd >= start and pd < folds[i + 1]:
                return f"fold{i + 1}"
        # If the patient ID is not within the range of any fold, return "test"
        return "Test"

def patients_timepoints(dataset_dir: str):
    """
    Get the count of timepoints for each patient in the dataset.

    Args:
        dataset_dir (str): The path to the dataset directory.

    Returns:
        dict: A dictionary containing the count of timepoints for each patient.

    Example:
        >>> from neuro_disease_detector.utils.utils_dataset import patients_timepoints
        >>> import os
        >>>
        >>> # Define the path to the dataset directory
        >>> dataset_dir = os.getcwd()
        >>>
        >>> timepoints = _patients_timepoints(dataset_dir)
        >>> print(timepoints)
        {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, ...}
    """
    
    # Define the base dataset path. By default, it's the "train" directory within the given dataset directory.
    dataset_path = f"{dataset_dir}/MSLesSeg-Dataset/train"

    # Initialize a dictionary to store the count of timepoints for each patient.
    timepoints = {}
    
    # Iterate through patient directories numbered from 1 to 53.
    for pd in range(1, 54):
        # Initialize the count of timepoints for the current patient to 0.
        timepoints[pd] = 0
        
        # Skip patient 30 as an exception (possibly due to missing or invalid data).
        if pd == 30:
            continue

        # Define the path for the current patient directory.
        pd_path = f"{dataset_path}/P{pd}"
        
        # Iterate through the potential timepoint directories (T1 to T4).
        for td in range(1, 5):
            # Check if the directory for the current timepoint exists. If not, exit the loop.
            if not os.path.exists(f"{pd_path}/T{td}"):
                break
            
            # Increment the timepoints count for the current patient.
            timepoints[pd] += 1

    # Return the dictionary containing the count of timepoints for each patient.
    return timepoints

def get_patient_by_test_id(test_id: int | str):
    """ 
    Given a test ID and a list with the number of tests per patient,
    return the patient to which the test belongs.

    Args:
        test_id (int | str): The ID of the test.

    Returns:
        str: The patient to which the test belongs.

    Example:
        >>> from neuro_disease_detector.utils.utils_dataset import get_patient_by_test_id
        >>>
        >>> # Define the test ID and the list of timepoints per patient
        >>> test_id = 3
        >>>
        >>> # Get the patient to which the test belongs
        >>> patient = get_patient_by_test_id(test_id)
        >>> print(patient)
        P1
    """

    timepoints_patient = [3,4,4,3,2,3,2,2,3,2,2,4,2,4,1,1,1,1,4,3,1,1,2,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2]
    
    test_id = int(test_id)
    current_id = 0

    for i, num_tests in enumerate(timepoints_patient):
        current_id += num_tests
        if test_id <= current_id:
            return f"P{i + 1}"
        
    return "ID not found"

def download_dataset_from_cloud(folder_name: str, url: str, extract_folder: bool = True) -> None:
    """
    Downloads and extracts a dataset from a cloud storage URL.
    
    Args:
        folder_name (str): The folder where the dataset will be extracted.
        url (str): The URL to download the dataset from.
    
    Returns:
        None

    Raises:
        FileNotFoundError: If the downloaded file cannot be found.
        zipfile.BadZipFile: If the ZIP file is invalid or corrupted.
    """

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
