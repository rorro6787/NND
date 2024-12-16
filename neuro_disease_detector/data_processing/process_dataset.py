import os
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from neuro_disease_detector.utils.utils_dataset import extract_contours_mask, load_nifti_image
from neuro_disease_detector.data_processing.extract_dataset import extract_dataset, download_dataset_from_cloud

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

dataset0 = "MSLesSeg-Dataset"
dataset1 = "MSLesSeg-Dataset-a"

def create_dataset_structure() -> None:
    """
    Creates a directory structure for a dataset with five folds, each containing
    subdirectories for images and labels. Additionally, it creates a 'masks' directory.

    The function will create the following directory structure:

    This will create the following directory structure:
        <dataset_path>/
            ├── fold1/
            │   ├── images/  # Subdirectory for image files
            │   └── labels/  # Subdirectory for label files
            ├── fold2/
            │   ├── images/  # Subdirectory for image files
            │   └── labels/  # Subdirectory for label files
            ├── fold3/
            │   ├── images/  # Subdirectory for image files
            │   └── labels/  # Subdirectory for label files
            ├── fold4/
            │   ├── images/  # Subdirectory for image files
            │   └── labels/  # Subdirectory for label files
            ├── fold5/
            │   ├── images/  # Subdirectory for image files
            │   └── labels/  # Subdirectory for label files
            └── masks/        # Subdirectory for mask files
    
    Args:
        None: The function uses a predefined base directory called `MSLesSeg-Dataset-a`.
    
    Example:
        create_dataset_structure()

    This will create the directory structure under the path `MSLesSeg-Dataset-a` as described above.

    """

    # Define the base dataset path
    dataset1_path = Path(dataset1)

    # Loop through each fold number from 1 to 5
    for i in range(1, 6):
        # Define the path for the current fold directory
        fold_path = dataset1_path / f"fold{i}"
        
        # Create 'images' and 'labels' subdirectories for the current fold
        (fold_path / "images").mkdir(parents=True, exist_ok=True)
        (fold_path / "labels").mkdir(parents=True, exist_ok=True)
    
    # Create a 'masks' directory at the base level
    (dataset1_path / "masks").mkdir(parents=True, exist_ok=True)
    logging.info(f"Base directory created: {dataset1_path}")

def process_training_dataset() -> None:
    """
    Processes the MRI dataset for training by organizing the images and labels into specific directories
    for training, testing, and validation. It iterates through patient directories and timepoints, 
    and stores the processed data in the appropriate folder structure.

    The function creates directories for training, testing, and validation if they do not exist.
    The data is split into images and labels and stored accordingly.
    
    Parameters:
    - None: This function does not accept any arguments. It uses predefined paths for datasets.

    Returns:
    - None: The function does not return any value, but it modifies the directory structure and organizes the dataset into appropriate folders.
    """

    # Define paths for the source training dataset and the destination dataset
    train0_path = Path(dataset0) / "train"

    # Create necessary folder structure for images and labels
    create_dataset_structure()

    # Iterate through each patient's directory in the training dataset
    for patient_dir in train0_path.iterdir():
        # Skip non-directory files
        if not patient_dir.is_dir():
            continue 
        # Get patient identifier
        patient_id = patient_dir.name
        logging.info(f"Patient {patient_id} being processed")

        # Iterate through each timepoint directory for the current patient
        for timepoint_dir in patient_dir.iterdir():
            # Skip non-directory files
            if not timepoint_dir.is_dir():
                continue  
            # Get timepoint identifier
            timepoint_id = timepoint_dir.name

            # Process the images and labels for the given patient and timepoint
            process_patient_timepoint(patient_id, timepoint_id)

def process_patient_timepoint(patient_id: str, timepoint_id: str) -> None:
    folder_path = Path(dataset0) / "train" / patient_id / timepoint_id

    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Load the mask file based on the expected naming convention
    mask = load_nifti_image(
        Path(folder_path) / f"{patient_id}_{timepoint_id}_MASK.nii"
    )

    # Extract slices from the mask image for comparison
    mask_slices = make_mask_slices(mask, patient_id, timepoint_id)
    
    # Loop through each file in the folder to process NIfTI images
    for file in files:
        # Skip non-NIfTI files and the mask file itself
        if not file.endswith(".nii") or file.endswith("_MASK.nii"):
            continue

        # Load the 3D image data from the current NIfTI file
        image = load_nifti_image(Path(folder_path) / file)

        file_name = file.split(".")[0]

        make_image_slices(image, mask_slices, patient_id, timepoint_id, file_name)

def make_mask_slices(mask: np.ndarray, patient_id: str, timepoint_id: str) -> list:
    """
    This function extracts sagittal, coronal, and axial slices from a 3D mask image and performs contour extraction
    for each slice. The extracted slices and their corresponding annotations (contours) are saved and returned.

    Parameters:
    ----------
    mask : np.ndarray
        A 3D numpy array representing the mask image (e.g., brain scan or other volumetric data).
    patient_id : str
        The ID of the patient. Used as part of the filename.

    timepoint_id : str
        The ID of the timepoint (e.g., the date or stage of the imaging). Used as part of the filename.

    Returns:
    -------
    sagittal_slices : list of annotations
        A list containing the annotations (contours) for each sagittal slice.
    coronal_slices : list of annotations
        A list containing the annotations (contours) for each coronal slice.
    axial_slices : list of annotations
        A list containing the annotations (contours) for each axial slice.
    """

    # Extract sagittal slices (along the first axis of the image)
    sagittal_slices = []
    for i in range(mask.shape[0]):
        sagittal_slice = mask[i, :, :]

        save_mask_slice(sagittal_slice, patient_id, timepoint_id, "sagittal", i)
        
        sagittal_slice_annotations = extract_contours_mask(sagittal_slice)
        sagittal_slices.append(sagittal_slice_annotations)

    # Extract coronal slices (along the second axis of the image)
    coronal_slices = []
    for i in range(mask.shape[1]):
        coronal_slice = mask[:, i, :]

        save_mask_slice(coronal_slice, patient_id, timepoint_id, "coronal", i)

        coronal_slice_annotations = extract_contours_mask(coronal_slice)
        coronal_slices.append(coronal_slice_annotations)

    # Extract axial slices (along the third axis of the image)
    axial_slices = []
    for i in range(mask.shape[2]):
        axial_slice = mask[:, :, i]

        save_mask_slice(coronal_slice, patient_id, timepoint_id, "axial", i)

        axial_slice_annotations = extract_contours_mask(axial_slice)
        axial_slices.append(axial_slice_annotations)

    return [sagittal_slices, coronal_slices, axial_slices]

def save_mask_slice(mask: np.ndarray, patient_id: str, timepoint_id: str, slice_type: str, index: int) -> None:
    """
    Saves the mask image as a PNG file with the name based on the patient ID, timepoint ID, slice type, and index.

    Parameters:
    - mask: np.ndarray
        A 2D NumPy array representing the binary mask image, where white pixels have a value of 255 and black pixels have a value of 0.

    - patient_id: str
        The ID of the patient. Used as part of the filename.

    - timepoint_id: str
        The ID of the timepoint (e.g., the date or stage of the imaging). Used as part of the filename.

    - slice_type: str
        The slice type (e.g., axial, coronal) of the image. Used as part of the filename.

    - index: int
        The index of the image in the dataset. Used as part of the filename.

    - dataset_dir: str
        The directory where the mask images will be saved. The function uses this path to save the mask image.
    
    Returns:
    - None
        This function does not return any value.
    """

    # Convert mask to uint8 to ensure proper format for saving
    mask = mask.astype(np.uint8)

    # Generate the image filename based on the modality, slice_type, and index
    image_name = f"{patient_id}_{timepoint_id}_{slice_type}_{index}"

    # Save the mask image in the specified directory
    plt.imsave(
        Path(dataset1) / "masks" / f"{image_name}.png",
        mask,
        cmap="gray"
    )

def make_image_slices(image: np.ndarray, mask_slices: list, patient_id: str, timepoint_id: str, file_name: str) -> None:
    """
    Extracts and saves slices from a 3D medical image along the sagittal, coronal, and axial planes, 
    along with their corresponding mask slices.

    Parameters:
    - image (np.ndarray): A 3D numpy array representing the medical image to be sliced.
    - mask_slices (list): A list containing three mask slices (one for each plane: sagittal, coronal, axial).
                          Each element should be a 2D array corresponding to the respective plane.
    - patient_id (str): A unique identifier for the patient whose data is being processed.
    - timepoint_id (str): A unique identifier for the time point of the scan.
    - file_name (str): The base file name used for saving the extracted slices.

    Behavior:
    - The function extracts slices from the 3D medical image along the sagittal, coronal, and axial planes.
    - For each slice, it also retrieves the corresponding mask slice from the `mask_slices` list.
    - Each extracted slice and its corresponding mask slice are saved using the `save_image_slice` function.
    - Slices are saved with the appropriate labeling based on the respective plane (sagittal, coronal, or axial) and the slice index.
    - The function does not return any value, as it performs the extraction and saving of slices directly.

    Notes:
    - The `mask_slices` list should contain three elements, where each element is a 2D array representing the mask for one of the three planes.
    - If further processing or additional functionality is needed, such as extracting specific regions of interest or performing image augmentation, this should be handled separately.
    """

    # Extract sagittal slices (along the first axis of the image)
    for i in range(image.shape[0]):
        sagittal_slice = image[i, :, :]
        sagittal_mask_slice = mask_slices[0][i]
        save_image_slice(sagittal_slice, sagittal_mask_slice, patient_id, "sagittal", file_name, i)

    # Extract coronal slices (along the second axis of the image)
    for i in range(image.shape[1]):
        coronal_slice = image[:, i, :]
        coronal_mask_slice = mask_slices[1][i]
        save_image_slice(coronal_slice, coronal_mask_slice, patient_id, "coronal", file_name, i)

    # Extract axial slices (along the third axis of the image)
    for i in range(image.shape[2]):
        axial_slice = image[:, i, :]
        axial_mask_slice = mask_slices[2][i]
        save_image_slice(axial_slice, axial_mask_slice, patient_id, "axial", file_name, i)

def save_image_slice(image_slice: np.ndarray, mask_slice: np.ndarray, patient_id: str, slice_type: str, file_name: str, index: int) -> None:
    """
    Saves a medical image slice and its corresponding mask to disk, processes the mask to extract white pixel coordinates, 
    and assigns the files to the appropriate dataset split (train, validation, or test).

    Parameters:
    - image_slice (np.ndarray): The 2D array representing the image slice to be saved.
    - mask_slice (np.ndarray): The 2D array representing the binary mask corresponding to the image slice.
    - patient_id (str): A unique identifier for the patient from whose scan the slice originates.
    - slice_type (str): The type of slice (e.g., axial, coronal, sagittal).
    - file_name (str): A base name used to uniquely identify the file.
    - index (int): The index of the slice within the scan, used for creating a unique file name.

    Function Behavior:
    - Assigns the image slice and its corresponding mask to either a training, test, or validation set, 
      with approximately 70% assigned to the training set, 15% to the test set, and 15% to the validation set.
    - Saves the image slice in the 'images' folder and the corresponding mask in the 'labels' folder 
      under the selected dataset split (train, test, or validation).
    - Extracts the coordinates of the white pixels from the mask (representing regions of interest) and saves them in 
      a text file in place of the mask image.
    - Removes the original mask image file to conserve disk space.

    File Paths:
    - Image slices are saved as PNG files in the `patients_dataset/[train/test/validation]/images/` directory.
    - Mask slices are saved as `.txt` files containing the white pixel coordinates in the `patients_dataset/[train/test/validation]/labels/` directory.

    Returns:
    - None: The function performs file I/O operations but does not return any value.
    """

    # Assign dataset split based on the patient_id (e.g., 'train', 'test', or 'validation')
    image_path, label_path = assign_dataset_split(patient_id)
    
    # Define a unique name for each image and mask based on the file name, slice type, and slice index
    image_name = f"{file_name}_{slice_type}_{index}"

    # Save the image slice as a PNG file in the corresponding directory (train/test/validation/images)
    plt.imsave(Path(image_path) / f"{image_name}.png", image_slice, cmap="gray")

    # Save the coordinates to the text file
    with open(Path(label_path) / f"{image_name}.txt", "w") as f:
        f.write(mask_slice) 

def assign_dataset_split(patient_id: str) -> Tuple[str, str]:
    """
    Assigns a patient to a specific dataset split (training, testing, or validation) 
    based on the patient's unique identifier. The patient ID is used to determine 
    the appropriate fold (subdivision) of the dataset.

    Parameters:
    - patient_id (str): The unique identifier of the patient.

    Returns:
    - Tuple[str, str]: A tuple containing:
        - The path to the directory containing the images for the assigned fold.
        - The path to the directory containing the labels for the assigned fold.

    The function determines the appropriate fold based on the numeric portion of the
    patient ID. Each fold corresponds to a specific range of patient numbers:
        - 'fold1' for patients 1-6
        - 'fold2' for patients 7-13
        - 'fold3' for patients 14-23
        - 'fold4' for patients 24-39
        - 'fold5' for patients 40-53

    The function assigns a fold based on the numeric portion of the patient ID. 
    Each fold corresponds to a specific range of patient IDs:
        - 'fold1' for patients with IDs 1-6
        - 'fold2' for patients with IDs 7-13
        - 'fold3' for patients with IDs 14-23
        - 'fold4' for patients with IDs 24-39
        - 'fold5' for patients with IDs 40-53

    Example:
        >>> image_path, label_path = assign_dataset_split(10)
        >>> print(image_path, label_path)
        '/path/to/dataset/fold1/images', '/path/to/dataset/fold1/labels'
    
    This will return the paths for the images and labels corresponding to 'fold1' 
    for a patient with ID 10.
    """

    patient_id = int(patient_id[1:])

    fold = ""

    if patient_id in range(1, 7):
        fold = "fold1"
    elif patient_id in range(7, 14):
        fold = "fold2"
    elif patient_id in range(14, 24):
        fold = "fold3"
    elif patient_id in range(24, 41):
        fold = "fold4"
    elif patient_id in range(41, 54):
        fold = "fold5"

    image_path = Path(dataset1) / fold / "images"
    label_path = Path(dataset1) / fold / "labels"

    return image_path, label_path

def process_dataset() -> bool:
    """
    Downloads and prepares the MSLesSeg-Dataset if it hasn't been extracted yet.
    
    The function checks if the dataset has already been extracted. If not, it calls 
    the `extract_dataset()` function to extract it. After extraction, it logs the process 
    and indicates whether the dataset is ready for use.
    
    Returns:
        bool: True if the dataset was successfully processed and is ready, False if an error occurred.
    """

    url = "https://drive.google.com/uc?export=download&id=1JD3Hb4U93EiRDVC4SNg4IVGqFJSfehPB"
    folder_name = "MSLesSeg-Dataset-a"

    # Check if the folder already exists
    if os.path.exists(folder_name):
        logging.info(f"The training dataset folder '{folder_name}' already exists.")
        return True

    # Check if dataset extraction is successful
    if not extract_dataset():
        return False
    
    try:
        # Extract the dataset (again, assuming this is required by the process)
        logging.info("Processing the training dataset...")
        download_dataset_from_cloud(url, folder_name)
        # process_training_dataset()
        logging.info("Training dataset processing finished.")
        return True
    except Exception as e:
        # Log any unexpected errors
        logging.error(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    process_dataset()
