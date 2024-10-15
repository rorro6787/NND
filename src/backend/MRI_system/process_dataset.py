# Standard imports
import os
from pathlib import Path
from enum import Enum
from typing import Tuple

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import gdown
import zipfile
import shutil

from scan import Scan

class SliceTypes(Enum):
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
    AXIAL = "axial"

def load_nifti_image(file_path: str) -> np.ndarray:
    """
    Loads a NIfTI (.nii) file and returns its 3D image data as a NumPy array.

    Parameters:
    file_path (str): Path to the NIfTI file.

    Returns:
    numpy.ndarray: 3D array of the image data.
    """

    # Load the NIfTI file
    img = nib.load(file_path)
    # Obtain image data as a 3D numpy array
    return img.get_fdata()


def process_training_dataset(base_path: str = os.getcwd()):
    """
    Processes the training dataset by iterating through all patients and their timepoints.
    It organizes and processes the MRI images and labels into respective directories for training, testing and validating.

    Parameters:
    - base_path (str): The root directory of the dataset, which contains subdirectories for patient data.

    Directory structure:
    - Creates the following directories in the current working directory:
        - 'patients_dataset/train/images': Stores training images.
        - 'patients_dataset/train/labels': Stores training labels.
        - 'patients_dataset/test/images': Stores test images.
        - 'patients_dataset/test/labels': Stores test labels.

    Returns: None
    """

    print("Processing the training dataset...")

    # Define paths for the original training dataset and the new dataset
    train_dataset_path = Path(os.path.join(base_path, "MSLesSeg-Dataset", "train"))
    new_dataset_path = Path(os.path.join(base_path, "MSLesSeg-Dataset-a"))

    create_fold_structure(new_dataset_path)

    # Iterate through each patient's directory in the training dataset
    for patient_dir in train_dataset_path.iterdir():
        if not patient_dir.is_dir():
            continue  # Skip non-directory entries
        patient_id = patient_dir.name

        # Iterate through each timepoint directory for the current patient
        for timepoint_dir in patient_dir.iterdir():
            if not timepoint_dir.is_dir():
                continue  # Skip non-directory entries
            timepoint_id = timepoint_dir.name
            scan = Scan(patient_id, timepoint_id)
            # Process the patient data for the current timepoint
            process_patient_timepoint(
                new_dataset_path,
                os.path.join(train_dataset_path, patient_id, timepoint_id),
                scan
            )

    print("Training dataset processing finished.")


def create_fold_structure(new_dataset_path: str) -> None:
    """
    Creates a directory structure for a dataset with five folds, each containing
    subdirectories for images and labels.

    Args:
        new_dataset_path (str): The path to the base directory where the fold
        directories will be created.

    The function performs the following actions:
    - It initializes a list of five fold directories named 'fold1' to 'fold5'
      within the specified base directory.
    - For each fold directory, it creates two subdirectories:
      - 'images': Intended to store image files.
      - 'labels': Intended to store corresponding label files.

    The `os.makedirs` function is used to create these directories, and the
    `exist_ok=True` parameter ensures that no error is raised if the
    directories already exist.

    Example:
        create_fold_structure('/path/to/dataset')

    This will create the following directory structure:
        /path/to/dataset/
            ├── fold1/
            │   ├── images/
            │   └── labels/
            ├── fold2/
            │   ├── images/
            │   └── labels/
            ├── fold3/
            │   ├── images/
            │   └── labels/
            ├── fold4/
            │   ├── images/
            │   └── labels/
            ├── fold5/
            │   ├── images/
            │   └── labels/
            └── test/
                ├── images/
                └── labels/

    """

    folds = [
        Path(os.path.join(new_dataset_path, "fold1")),
        Path(os.path.join(new_dataset_path, "fold2")),
        Path(os.path.join(new_dataset_path, "fold3")),
        Path(os.path.join(new_dataset_path, "fold4")),
        Path(os.path.join(new_dataset_path, "fold5")),
        Path(os.path.join(new_dataset_path, "test")),
    ]

    for fold in folds:
        os.makedirs(os.path.join(fold, "images"), exist_ok=True)
        os.makedirs(os.path.join(fold, "labels"), exist_ok=True)


def process_patient_timepoint(
    new_dataset_path: str, folder_path: str, scan: Scan
):
    """
    Processes all NIfTI (.nii) files in the specified folder, identifies the mask file, and compares it with the other images in the folder.

    Parameters:
    - new_dataset_path (str): The path to the new dataset directory where processed images will be saved.
    - folder_path (str): Path to the folder containing the NIfTI files (.nii).
    - scan (Scan): An instance of the Scan class containing patient and timepoint information.

    Function behavior:
    - The function lists all files in the given folder.
    - It identifies the mask file, which is expected to follow the naming convention '{patient_id}_{timepoint_id}_MASK.nii'.
    - For each remaining NIfTI file (excluding the mask), the function loads the 3D image data and compares it with the mask by calling `compare_image_with_mask`.

    Output:
    - Calls `compare_image_with_mask` for each non-mask NIfTI file, processing and saving the image slices and masks.

    Returns: None
    """

    # List all files in the specified folder
    files = os.listdir(folder_path)

    # Load the mask file based on the expected naming convention
    mask_image = load_nifti_image(
        os.path.join(folder_path, f"{scan.patient}_{scan.timespan}_MASK.nii")
    )

    # Extract slices from the mask image for comparison
    mask_sagittal, mask_coronal, mask_axial = make_slices(mask_image, mask=True)
    mask_slices = [mask_sagittal, mask_coronal, mask_axial]

    # Loop through each file in the folder to process NIfTI images
    for file in files:
        # Skip non-NIfTI files and the mask file itself
        if not file.endswith(".nii") or file.endswith("_MASK.nii"):
            continue

        # Load the 3D image data from the current NIfTI file
        image_data = load_nifti_image(os.path.join(folder_path, file))

        # Extract the file name without the extension (used to identify the image type)
        file_name = file.split(".")[0]

        scan.set_modality(file_name)

        # Compare the loaded image with the mask
        compare_image_with_mask(
            new_dataset_path, image_data, mask_slices, scan
        )


def make_slices(image: np.ndarray, mask: bool = False):
    """
    Extracts slices from a 3D medical image along sagittal, coronal, and axial planes.

    Parameters:
    - image (np.ndarray): A 3D numpy array representing the medical image.
    - mask (bool): A flag indicating whether to extract white pixel coordinates from the slices.
                   If True, the function will process the slices to extract white pixel coordinates.

    Returns:
    - sagittal_slices (list): A list of sagittal slices extracted from the image.
    - coronal_slices (list): A list of coronal slices extracted from the image.
    - axial_slices (list): A list of axial slices extracted from the image.

    Behavior:
    - The function extracts slices along the three principal planes (sagittal, coronal, axial).
    - If the `mask` parameter is set to True, it applies the `extract_white_pixel_coordinates_mask` function
      to each slice to obtain the white pixel coordinates instead of the raw slice data.
    """

    # (IT WILL CHANGE!!!!!!)

    # Extract sagittal slices (along the first axis of the image)
    sagittal_slices = []
    for i in range(image.shape[0]):
        sagittal_slice = image[i, :, :]
        if mask:
            # Extract white pixel coordinates from the sagittal slice if mask is True
            sagittal_slice = extract_contours_mask(sagittal_slice)
        sagittal_slices.append(sagittal_slice)

    # Extract coronal slices (along the second axis of the image)
    coronal_slices = []
    for i in range(image.shape[1]):
        coronal_slice = image[:, i, :]
        if mask:
            # Extract white pixel coordinates from the coronal slice if mask is True
            coronal_slice = extract_contours_mask(coronal_slice)
        coronal_slices.append(coronal_slice)

    # Extract axial slices (along the third axis of the image)
    axial_slices = []
    for i in range(image.shape[2]):
        axial_slice = image[:, :, i]
        if mask:
            # Extract white pixel coordinates from the axial slice if mask is True
            axial_slice = extract_contours_mask(axial_slice)
        axial_slices.append(axial_slice)

    return sagittal_slices, coronal_slices, axial_slices


def compare_image_with_mask(
    new_dataset_path: str,
    image: np.ndarray,
    mask_slices: list,
    scan: Scan
):
    """
    Compares a 3D medical image with its corresponding mask by slicing the image along different planes
    (sagittal, coronal, axial) and saves the results.

    Parameters:
    - new_dataset_path (str): The path to the new dataset directory where slices will be saved.
    - image (np.ndarray): A 3D numpy array representing the medical image.
    - mask_slices (list of np.ndarray): A list containing 3D numpy arrays representing the mask images for each plane.
    - scan (Scan): An instance of the Scan class containing patient, timepoint and modality information.

    Function behavior:
    - The function slices both the 3D image and its corresponding mask along the sagittal, coronal, and axial planes.
    - For each plane, the corresponding slices from both the image and mask are paired and collected.
    - The paired slices are then saved using the `save_image_slices` function for each of the three orientations (sagittal, coronal, and axial).

    Output:
    - Calls the `save_image_slices` function to save the slices and masks for each plane orientation (sagittal, coronal, axial).

    Returns: None
    """

    # Extract slices for each plane (sagittal, coronal, axial) from the 3D image
    sagittal_slices, coronal_slices, axial_slices = make_slices(image)

    # Save the slices and corresponding masks for each orientation (IT WILL CHANGE!!!!!!)
    save_image_slices(
        new_dataset_path,
        sagittal_slices,
        mask_slices[0],
        scan,
        SliceTypes.SAGITTAL,
    )
    save_image_slices(
        new_dataset_path,
        coronal_slices,
        mask_slices[1],
        scan,
        SliceTypes.CORONAL,
    )
    save_image_slices(
        new_dataset_path,
        axial_slices,
        mask_slices[2],
        scan,
        SliceTypes.AXIAL,
    )


def save_image_slices(
    new_dataset_path: str,
    image_slices: list,
    mask_slices: list,
    scan: Scan,
    slice_type: SliceTypes,
):
    """
    Saves image slices and corresponding masks to disk, and processes the masks to extract white pixel coordinates.

    Parameters:
    - new_dataset_path (str): The path to the new dataset directory where images and masks will be saved.
    - image_slices (list of tuples): A list where each tuple contains an image slice (numpy array).
    - mask_slices (list of tuples): A list where each tuple contains a corresponding mask (numpy array) for each image slice.
    - scan (Scan): An instance of the Scan class containing patient, timepoint and modality information.
    - slice_type (SliceTypes): Enum representing the type of slice (e.g., axial, coronal).

    Function behavior:
    - Randomly assigns each image slice and mask to either a training, test or validation set, with an approximately 15% chance of being assigned to the test set and another 15% to validation set.
    - Saves the image slices in the 'images' folder and the corresponding masks in the 'labels' folder within either the 'train' or 'test' directories, depending on the random assignment.
    - Extracts the coordinates of the white pixels from each mask (where white pixels represent specific features of interest) and saves them as text files in the same directory as the masks.
    - After extracting white pixel coordinates, the mask image file is removed to conserve storage.

    Output:
    - Image slices are saved in 'patients_dataset/[train/test]/images' and corresponding masks are saved in 'patients_dataset/[train/test]/labels'.
    - A text file containing the white pixel coordinates is saved in place of the mask image file.

    Returns: None
    """

    for i in range(len(image_slices)):
        img = image_slices[i]
        mask = mask_slices[i]

        output_image_path, output_label_path = assign_dataset_split(
            new_dataset_path, scan
        )

        # Define a unique name for each image and mask based on patient info, image type, and slice index
        image_name = f"{scan.modality}_{slice_type.value}_{i}"

        # Save the image slice as a PNG file
        plt.imsave(
            os.path.join(output_image_path, f"{image_name}.png"), img, cmap="gray"
        )

        # Extract white pixel coordinates from the mask and save them as a text file
        with open(os.path.join(output_label_path, f"{image_name}.txt"), "w") as f:
            f.write(mask)  # Save the coordinates to the text file


def assign_dataset_split(new_dataset_path: str, scan: Scan) -> Tuple[str, str]:
    """
    Assigns a patient to a specific dataset split (training, testing, or validation)
    based on the patient's unique identifier.

    Parameters:
    - new_dataset_path (str): The base path to the dataset directory.
    - scan (Scan): An instance of the Scan class containing patient, timepoint and modality information.

    Returns:
    - Tuple[str, str]: A tuple containing:
        - str: The path to the directory for the output images for the assigned fold.
        - str: The path to the directory for the output labels for the assigned fold.

    The function determines the appropriate fold based on the numeric portion of the
    patient ID. Each fold corresponds to a specific range of patient numbers:
        - 'fold1' for patients 1-6
        - 'fold2' for patients 7-13
        - 'fold3' for patients 14-23
        - 'fold4' for patients 24-39
        - 'fold5' for patients 40-53

    Example:
        output_images, output_labels = assign_dataset_split('/path/to/dataset', 'P10')

    This will return the paths for the images and labels corresponding to 'fold1':
        ('/path/to/dataset/fold1/images', '/path/to/dataset/fold1/labels')
    """

    patient_number = int(scan.patient[1:])

    fold = ""

    if patient_number in range(1, 6):
        fold = "fold1"
    elif patient_number in range(6, 12):
        fold = "fold2"
    elif patient_number in range(12, 19):
        fold = "fold3"
    elif patient_number in range(19, 28):
        fold = "fold4"
    elif patient_number in range(28, 41):
        fold = "fold5"
    else:
        fold = "test"

    output_image_path = os.path.join(new_dataset_path, fold, "images")
    output_label_path = os.path.join(new_dataset_path, fold, "labels")

    return output_image_path, output_label_path


def extract_contours_mask(mask: np.ndarray) -> str:
    """
    Extracts the normalized coordinates of white pixels from a black and white image
    represented as a NumPy array. The output format is a string starting with '0',
    followed by normalized coordinates of white pixels (where the pixel value is 255).

    Each coordinate is normalized by the image dimensions, yielding values between 0 and 1.
    The coordinates are formatted to six decimal places.

    Parameters:
    - mask: np.ndarray
        A 2D numpy array representing the black and white image, where white pixels
        have a value of 255 and black pixels have a value of 0.

    Returns: str
        A string containing the normalized coordinates of contours found in the image in the format:
        "0 <x_center> <y_center> <width> <height> <x1> <y1> <x2> <y2> ... <xn> <yn>"
        for each contour. Each coordinate is normalized to fall between 0 and 1,
        and the values are formatted to six decimal places.
    """

    # Get the dimensions of the image
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_height, mask_width = mask.shape

    annotations = ""
    for contour in contours:
        # Get the bounding box of each object
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the center of the bounding box
        x_center = (x + w / 2) / mask_width
        y_center = (y + h / 2) / mask_height

        # Normalize the width and height
        width = w / mask_width
        height = h / mask_height

        # Write the class (assuming class 0) and the bounding box
        annotations += f"0 {x_center} {y_center} {width} {height}"

        # Add the points of the contour (normalized)
        for point in contour:
            px, py = point[0]
            annotations += f" {px/mask_width} {py/mask_height}"

        # Add a new line for each object
        annotations += "\n"

    return annotations


def download_mslesseg_dataset() -> str:
    """
    Downloads and extracts the MSLesSeg Dataset from Google Drive if it is not already present
    in the current working directory.

    The function checks if the dataset folder already exists. If it does not, it downloads a
    ZIP file containing the dataset from a specified Google Drive URL, extracts the contents,
    and then deletes the ZIP file to save space. If the dataset is already downloaded,
    it simply informs the user.

    Steps:
    1. Get the current working directory.
    2. Check if the dataset directory exists.
    3. If the directory does not exist:
        a. Specify the Google Drive URL to download the dataset.
        b. Set the name for the ZIP file to be saved locally.
        c. Download the ZIP file from the URL using `gdown`.
        d. Extract the contents of the ZIP file into the current directory.
        e. Remove the ZIP file after extraction to clean up.
    4. If the directory exists, inform the user that the dataset is already downloaded.

    Returns:
        str: The path to the dataset directory.
    """

    current_directory = os.getcwd()  # Get the current working directory

    # Name of the ZIP file to save locally
    zip_file_name = "MSLesSeg-Dataset.zip"
    dataset_folder_name = "MSLesSeg-Dataset"

    # Check if dataset directory exists
    dataset_directory_path = os.path.join(current_directory, dataset_folder_name)
    if not os.path.exists(dataset_directory_path):
        # URL to download the zip file from Google Drive
        google_drive_url = "https://drive.google.com/uc?export=download&id=1y55uyeo79M4Cw6eg_G9-06qU6jhJphsM"

        # Download the zip file from the URL
        print("Downloading dataset...")
        gdown.download(
            google_drive_url, zip_file_name, quiet=False
        )  # Download using gdown
        print(f"File downloaded as {zip_file_name}")

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
            zip_ref.extractall(current_directory)  # Extract to current directory

        print("Dataset downloaded and extracted.")
        # Delete the ZIP file after extraction
        os.remove(zip_file_name)
        print("Dataset downloading process finished.")

    else:
        print(
            "Dataset already downloaded in the system..."
        )  # Inform user if dataset exists

    # Prepare the dataset with the function defined below
    prepare_dataset(dataset_directory_path)


def prepare_dataset(dataset_path: str):
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

    The function does not return any value. Instead, it modifies
    the file system by deleting files.

    Steps:
    1. Define the path to the training dataset directory.
    2. List all directories within the training dataset, which correspond to different patients.
    3. For each patient directory:
       - List all timepoint directories.
       - For each timepoint directory:
           - List all files within the timepoint directory.
           - For each file:
               - Check if it has a '.gz' extension.
               - If a decompressed version (same name without the '.gz' suffix) exists, add the original
                 compressed file to a set of files to delete.
               - If the decompressed file does not exist, decompress the '.gz' file using the
                 `gunzip` command.
    4. Print out the files that will be deleted and then proceed to remove them from the file system.

    Raises:
        OSError: If there are issues with file or directory access during the execution.
    """

    print("Processing the dataset...")
    # Step 1: Define the path to the training dataset directory
    training_directory = Path(os.path.join(dataset_path, "train"))

    # Step 2: Initialize a set to keep track of files to delete
    files_to_remove = set()

    # Step 3: Loop through each patient directory
    for patient_directory in training_directory.iterdir():
        if not patient_directory.is_dir():
            continue
        if patient_directory.name == "P30":
            # We delete the patient P30 because it has a corrupted structure
            shutil.rmtree(patient_directory)
            continue
        # Loop through each timepoint directory
        for timepoint_directory in patient_directory.iterdir():
            if not timepoint_directory.is_dir():
                continue
            # List all files within the timepoint directory
            files_in_timepoint = os.listdir(timepoint_directory)

            # Loop through each file in the timepoint directory
            for file_name in files_in_timepoint:
                # Check if the file has a '.gz' extension
                if not file_name.endswith(".gz"):
                    continue
                # Define the name of the decompressed file (removing the '.gz' suffix)
                decompressed_file_name = file_name[:-3]
                decompressed_file_path = timepoint_directory / decompressed_file_name

                # Check if the decompressed file already exists
                if decompressed_file_path.is_file():
                    # If it exists, add the original compressed file to the delete list
                    files_to_remove.add(timepoint_directory / file_name)
                else:
                    # If not, decompress the '.gz' file using the gunzip command
                    os.system(f"gunzip {timepoint_directory / file_name}")

    # Step 4: Print out the files that will be deleted
    if files_to_remove:
        # Remove the files marked for deletion from the filesystem
        for file_path in files_to_remove:
            os.remove(file_path)
    else:
        print("No files to delete.")

    print("Dataset processing finished.")


def patients_timepoints(
    dataset_path: str = os.path.join(os.getcwd(), "MSLesSeg-Dataset"),
):
    """
    Counts the number of timepoints for each patient in the dataset.

    This function traverses through the training directory within the given dataset path
    and counts how many timepoints (subdirectories) exist for each patient (subdirectory).
    It skips non-directory files.

    Parameters:
    -----------
    dataset_path : str, optional
        The path to the dataset directory. Defaults to the 'MSLesSeg-Dataset/train'
        directory located in the current working directory.

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

    training_directory = Path(os.path.join(dataset_path, "train"))
    patients = {}
    for patient_directory in training_directory.iterdir():
        if not patient_directory.is_dir():
            continue
        patients[patient_directory.name] = 0
        for timepoint_directory in patient_directory.iterdir():
            if not timepoint_directory.is_dir():
                continue
            patients[patient_directory.name] += 1
    return patients


if __name__ == "__main__":
    # process_training_dataset(os.path.join(os.getcwd(), 'MSLesSeg-Dataset'))
    # download_mslesseg_dataset()
    # extract_white_pixel_coordinates(os.path.join(os.getcwd(), 'patients_dataset', 'P1', 'T1', 't1', '108', 'mask.png'), os.path.join(os.getcwd(), 'test.txt'))
    # process_training_dataset(os.getcwd())

    download_mslesseg_dataset()
    process_training_dataset()
