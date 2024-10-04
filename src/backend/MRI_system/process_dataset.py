import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import os
import gdown
import zipfile
from PIL import Image

from enum import Enum

class ImageTypes(Enum):
    FLAIR = "flair"
    MASK = "mask"
    T1 = "t1"
    T2 = "t2"

def load_nifti_image(file_path: str):
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

def process_patient_timepoint(folder_path: str, patient_id: str, timepoint_id: str):
    """
    Processes all .nii files in the specified folder. Identifies one as a mask file and compares each of the remaining images with the mask.

    Parameters:
    folder_path (str): Path to the folder containing NIfTI files.

    Returns:
    None
    """

    # List all files in the folder
    files = os.listdir(folder_path)
    images_3D = []
    mask_image = None
    
    # Loop through each file in the folder
    for file in files:  
        # Check if the file is a NIfTI file
        if file.endswith('.nii'):
            # Load the 3D image data from the file
            image_data = load_nifti_image(os.path.join(folder_path, file))
            
            # Check if the file is the mask file
            if file.endswith('MASK.nii'):
                mask_image = image_data  # Set the mask
            else:
                # Append other scans to the list
                images_3D.append(image_data)
    
    # Compare each scan with the mask
    compare_image_with_mask(images_3D[0], mask_image, patient_id, timepoint_id, ImageTypes.FLAIR)
    compare_image_with_mask(images_3D[1], mask_image, patient_id, timepoint_id, ImageTypes.T1)
    compare_image_with_mask(images_3D[2], mask_image, patient_id, timepoint_id, ImageTypes.T2)

def compare_image_with_mask(image: np.ndarray, mask: np.ndarray, patient_id: str, timepoint_id: str, image_type: ImageTypes):
    """
    Compares a 3D image with a mask and saves the slices.

    Parameters:
    image (np.ndarray): The 3D image to compare.
    mask (np.ndarray): The mask image.
    patient_id (str): Patient identifier.
    timepoint_id (str): Timepoint identifier.
    image_type (ImageTypes): Type of the image (FLAIR, T1, T2).
    
    Returns:
    None
    """

    sagittal_slices = []
    for i in range(image.shape[0]):
        sagittal_slice = image[i, :, :]
        mask_slice = mask[i, :, :]
        sagittal_slices.append((sagittal_slice, mask_slice))
    
    coronal_slices = []
    for i in range(image.shape[1]):
        coronal_slice = image[:, i, :]
        mask_slice = mask[:, i, :]
        coronal_slices.append((coronal_slice, mask_slice))

    axial_slices = []
    for i in range(image.shape[2]):
        axial_slice = image[:, :, i]
        mask_slice = mask[:, :, i]
        axial_slices.append((axial_slice, mask_slice))
    
    # Save the image slices
    save_image_slices(sagittal_slices, patient_id, timepoint_id, image_type)
    save_image_slices(coronal_slices, patient_id, timepoint_id, image_type)
    save_image_slices(axial_slices, patient_id, timepoint_id, image_type)

def save_image_slices(image_slices, patient_id: str, timepoint_id: str, image_type: ImageTypes):
    """
    Saves the image slices and their corresponding masks.

    Parameters:
    image_slices (list): List of tuples containing image and mask slices.
    patient_id (str): Patient identifier.
    timepoint_id (str): Timepoint identifier.
    image_type (ImageTypes): Type of the image (FLAIR, T1, T2).
    
    Returns:
    None
    """

    output_path = os.path.join(os.getcwd(), 'patients_dataset')

    for i, (img, mask) in enumerate(image_slices):
        image_dir = os.path.join(output_path, patient_id, timepoint_id, image_type.value, str(i))
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        plt.imsave(os.path.join(image_dir, "image.png"), img, cmap='gray')
        plt.imsave(os.path.join(image_dir, "mask.png"), mask, cmap='gray')

def process_training_dataset(base_path: str):
    """
    Processes the training dataset for all patients and timepoints.

    Parameters:
    base_path (str): Path to the dataset.

    Returns:
    None
    """

    print("Processing the training dataset...")

    train_dataset_path = Path(os.path.join(base_path, 'MSLesSeg-Dataset', 'train'))
    for patient_dir in train_dataset_path.iterdir():
        if patient_dir.is_dir():
            patient_id = patient_dir.name
            for timepoint_dir in patient_dir.iterdir():
                if timepoint_dir.is_dir():
                    timepoint_id = timepoint_dir.name
                    process_patient_timepoint(os.path.join(train_dataset_path, patient_id, timepoint_id), patient_id, timepoint_id)

    print("Training dataset processing finished.")

def extract_white_pixel_coordinates(image_path, output_path=None):
    """
    Extracts the coordinates of white pixels from a black and white image 
    and writes them to a text file in the format:
    0 <x1> <y1> <x2> <y2> ... <xn> <yn>.

    Parameters:
    - image_path: str, the path to the input black and white image file.
    - output_path: str, the path to the output text file (optional).
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert the image to black and white (mode '1' for 1-bit pixels)
    img = img.convert('1')  
    
    # Get the dimensions of the image
    width, height = img.size
    
    # List to store the coordinates of white pixels
    coordinates = []
    
    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the pixel value
            pixel = img.getpixel((x, y))
            # Check if the pixel is white (255 in mode '1')
            if pixel == 255:  
                # Normalize and format with six decimal places
                coordinates.append(f"{x/width:.6f} {y/height:.6f}")
    
    # Format the output string in the required format
    result = "0 " + " ".join(coordinates)
    
    # Optionally write the result to a text file if an output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(result)

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
        if patient_directory.is_dir():            
            # Loop through each timepoint directory
            for timepoint_directory in patient_directory.iterdir():
                if timepoint_directory.is_dir():
                    # List all files within the timepoint directory
                    files_in_timepoint = os.listdir(timepoint_directory)
                    
                    # Loop through each file in the timepoint directory
                    for file_name in files_in_timepoint:
                        print(f"\t\tFile: {file_name}")
                        
                        # Check if the file has a '.gz' extension
                        if file_name.endswith('.gz'):
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
        print(f"Files to delete: {files_to_remove}")
        
        # Remove the files marked for deletion from the filesystem
        for file_path in files_to_remove:
            os.remove(file_path)
    else:
        print("No files to delete.")
    
    print("Dataset processing finished.")


if __name__ == "__main__":
    # process_training_dataset(os.path.join(os.getcwd(), 'MSLesSeg-Dataset'))
    # download_mslesseg_dataset()
    # process_training_dataset(os.getcwd())
    extract_white_pixel_coordinates(os.path.join(os.getcwd(), 'patients_dataset', 'P1', 'T1', 't1', '108', 'mask.png'), os.path.join(os.getcwd(), 'test.txt'))
