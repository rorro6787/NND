import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import os
import gdown
import zipfile
import shutil
from enum import Enum

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
    It organizes and processes the MRI images and labels into respective directories for training and testing.

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
    train_dataset_path = Path(os.path.join(base_path, 'MSLesSeg-Dataset', 'train'))
    new_dataset_path = Path(os.path.join(base_path, 'MSLesSeg-Dataset-a'))

    # Create necessary directories for storing training and testing images and labels
    os.makedirs(os.path.join(os.getcwd(), new_dataset_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), new_dataset_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), new_dataset_path, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), new_dataset_path, 'test', 'labels'), exist_ok=True)

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
            
            # Process the patient data for the current timepoint
            process_patient_timepoint(
                new_dataset_path,
                os.path.join(train_dataset_path, patient_id, timepoint_id),
                patient_id,
                timepoint_id
            )

    print("Training dataset processing finished.")

def process_patient_timepoint(new_dataset_path: str, folder_path: str, patient_id: str, timepoint_id: str):
    """
    Processes all NIfTI (.nii) files in the specified folder, identifies the mask file, and compares it with the other images in the folder.

    Parameters:
    - new_dataset_path (str): The path to the new dataset directory where processed images will be saved.
    - folder_path (str): Path to the folder containing the NIfTI files (.nii).
    - patient_id (str): Unique identifier for the patient.
    - timepoint_id (str): Identifier for the timepoint of image acquisition.

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
    mask_image = load_nifti_image(os.path.join(folder_path, f"{patient_id}_{timepoint_id}_MASK.nii"))
    
    # Extract slices from the mask image for comparison
    mask_sagittal, mask_coronal, mask_axial = make_slices(mask_image, mask=True)
    mask_slices = [mask_sagittal, mask_coronal, mask_axial]
    
    # Loop through each file in the folder to process NIfTI images
    for file in files:  
        # Skip non-NIfTI files and the mask file itself
        if not file.endswith('.nii') or file.endswith('_MASK.nii'):
            continue

        # Load the 3D image data from the current NIfTI file
        image_data = load_nifti_image(os.path.join(folder_path, file))
        
        # Extract the file name without the extension (used to identify the image type)
        file_name = file.split('.')[0]

        # Compare the loaded image with the mask
        compare_image_with_mask(new_dataset_path, image_data, mask_slices, patient_id, timepoint_id, file_name)

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
            sagittal_slice = extract_white_pixel_coordinates_mask(sagittal_slice)
        sagittal_slices.append(sagittal_slice)
    
    # Extract coronal slices (along the second axis of the image)
    coronal_slices = []
    for i in range(image.shape[1]):
        coronal_slice = image[:, i, :]
        if mask:
            # Extract white pixel coordinates from the coronal slice if mask is True
            coronal_slice = extract_white_pixel_coordinates_mask(coronal_slice)
        coronal_slices.append(coronal_slice)

    # Extract axial slices (along the third axis of the image)
    axial_slices = []
    for i in range(image.shape[2]):
        axial_slice = image[:, :, i]
        if mask:
            # Extract white pixel coordinates from the axial slice if mask is True
            axial_slice = extract_white_pixel_coordinates_mask(axial_slice)
        axial_slices.append(axial_slice)

    return sagittal_slices, coronal_slices, axial_slices
    
def compare_image_with_mask(new_dataset_path: str, image: np.ndarray, mask_slices: list, patient_id: str, timepoint_id: str, image_type: str):
    """
    Compares a 3D medical image with its corresponding mask by slicing the image along different planes
    (sagittal, coronal, axial) and saves the results.

    Parameters:
    - new_dataset_path (str): The path to the new dataset directory where slices will be saved.
    - image (np.ndarray): A 3D numpy array representing the medical image.
    - mask_slices (list of np.ndarray): A list containing 3D numpy arrays representing the mask images for each plane.
    - patient_id (str): Unique identifier for the patient.
    - timepoint_id (str): Identifier for the timepoint of image acquisition.
    - image_type (ImageTypes): Enum indicating the type of image (e.g., FLAIR, T1, T2).

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
    save_image_slices(new_dataset_path, sagittal_slices, mask_slices[0], patient_id, timepoint_id, image_type, SliceTypes.SAGITTAL)
    save_image_slices(new_dataset_path, coronal_slices, mask_slices[1], patient_id, timepoint_id, image_type, SliceTypes.CORONAL)
    save_image_slices(new_dataset_path, axial_slices, mask_slices[2], patient_id, timepoint_id, image_type, SliceTypes.AXIAL)

def save_image_slices(new_dataset_path: str, image_slices: list, mask_slices: list, patient_id: str, timepoint_id: str, image_type: str, slice_type: SliceTypes):
    """
    Saves image slices and corresponding masks to disk, and processes the masks to extract white pixel coordinates.

    Parameters:
    - new_dataset_path (str): The path to the new dataset directory where images and masks will be saved.
    - image_slices (list of tuples): A list where each tuple contains an image slice (numpy array).
    - mask_slices (list of tuples): A list where each tuple contains a corresponding mask (numpy array) for each image slice.
    - patient_id (str): A unique identifier for the patient.
    - timepoint_id (str): Identifier for the image acquisition timepoint.
    - image_type (ImageTypes): Enum representing the type of image (e.g., FLAIR, T1, T2).
    - slice_type (SliceTypes): Enum representing the type of slice (e.g., axial, coronal).

    Function behavior:
    - Randomly assigns each image slice and mask to either a training or test set, with an approximately 20% chance of being assigned to the test set.
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
        
        # Randomly assign to test set with approximately 20% probability
        prop_test = np.random.randint(1, 10)

        # Determine output paths based on random test/train assignment
        if prop_test < 3:
            output_image_path = os.path.join(new_dataset_path, 'test', 'images')
            output_label_path = os.path.join(new_dataset_path, 'test', 'labels')
        else:
            output_image_path = os.path.join(new_dataset_path, 'train', 'images')
            output_label_path = os.path.join(new_dataset_path, 'train', 'labels')

        # Define a unique name for each image and mask based on patient info, image type, and slice index
        image_name = f"{patient_id}_{timepoint_id}_{image_type}_{slice_type.value}_{i}"
        
        # Save the image slice as a PNG file
        plt.imsave(os.path.join(output_image_path, f"{image_name}.png"), img, cmap='gray')
        
        # Extract white pixel coordinates from the mask and save them as a text file
        with open(os.path.join(output_label_path, f"{image_name}.txt"), 'w') as f:
            f.write(mask)  # Save the coordinates to the text file

def extract_white_pixel_coordinates_mask(image_array: np.ndarray) -> str:
    """
    Extracts the normalized coordinates of white pixels from a black and white image
    represented as a NumPy array. The output format is a string starting with '0',
    followed by normalized coordinates of white pixels (where the pixel value is 255).
    
    Each coordinate is normalized by the image dimensions, yielding values between 0 and 1.
    The coordinates are formatted to six decimal places.
    
    Parameters:
    - image_array: np.ndarray
        A 2D numpy array representing the black and white image, where white pixels
        have a value of 1 and black pixels have a value of 0.

    Returns: str
        A string containing the normalized coordinates of white pixels in the format:
        "0 <x1> <y1> <x2> <y2> ... <xn> <yn>".
    """

    # Get the dimensions of the image
    height, width = image_array.shape
    
    # List to store the normalized coordinates of white pixels
    coordinates = []
    
    # Iterate over each pixel in the array
    for y in range(height):
        for x in range(width):
            # Check if the pixel is white (1)
            if image_array[y, x] == 1:  
                # Normalize the coordinates and format them with six decimal places
                coordinates.append(f"{x/width:.6f} {y/height:.6f}")
    
    # Format the output string in the required format
    result = "0 " + " ".join(coordinates)
    
    # Return the result string
    return result

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
    zip_file_name = 'MSLesSeg-Dataset.zip'
    dataset_folder_name = 'MSLesSeg-Dataset'

    # Check if dataset directory exists
    dataset_directory_path = os.path.join(current_directory, dataset_folder_name)
    if not os.path.exists(dataset_directory_path):
        # URL to download the zip file from Google Drive
        google_drive_url = "https://drive.google.com/uc?export=download&id=1y55uyeo79M4Cw6eg_G9-06qU6jhJphsM"

        # Download the zip file from the URL
        print("Downloading dataset...")
        gdown.download(google_drive_url, zip_file_name, quiet=False)  # Download using gdown
        print(f"File downloaded as {zip_file_name}")

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(current_directory)  # Extract to current directory
        
        print("Dataset downloaded and extracted.")
        # Delete the ZIP file after extraction
        os.remove(zip_file_name)
        print("Dataset downloading process finished.")

    else:
        print("Dataset already downloaded in the system...")  # Inform user if dataset exists

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
    training_directory = Path(os.path.join(dataset_path, 'train'))

    # Step 2: Initialize a set to keep track of files to delete
    files_to_remove = set()
        
    # Step 3: Loop through each patient directory
    for patient_directory in training_directory.iterdir():
        if not patient_directory.is_dir():   
            continue   
        if patient_directory.name == 'P30':   
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
                if not file_name.endswith('.gz'):
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

if __name__ == "__main__":
    # process_training_dataset(os.path.join(os.getcwd(), 'MSLesSeg-Dataset'))
    # download_mslesseg_dataset()
    # extract_white_pixel_coordinates(os.path.join(os.getcwd(), 'patients_dataset', 'P1', 'T1', 't1', '108', 'mask.png'), os.path.join(os.getcwd(), 'test.txt'))
    # process_training_dataset(os.getcwd())
    
    # download_mslesseg_dataset()
    process_training_dataset()            

    
