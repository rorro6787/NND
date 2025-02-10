import matplotlib.pyplot as plt
import numpy as np
import os 

from neuro_disease_detector.yolo.utils.utils_nifti import extract_contours_mask, load_nifti_image
from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud, split_assign
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

def yolo_init():
    """
    Initialize the YOLO dataset processing pipeline.

    Args:
        None

    Returns:
        None

    Example:
        >>> from neuro_disease_detector.yolo.process_dataset import yolo_init
        >>>
        >>> # Initialize the YOLO dataset processing pipeline
        >>> nnUNet_init(dataset_id, configuration, fold, trainer)
    """

    dataset_dir = f"{cwd}/MSLesSeg-Dataset"
    yolo_dataset = f"{cwd}/MSLesSeg-Dataset-a"

    logger.info(f"Downloading MSLesSeg-Dataset for yolo pipeline...")
    url = "https://drive.google.com/uc?export=download&id=1A3ZpXHe-bLpaAI7BjPTSkZHyQwEP3pIi"
    download_dataset_from_cloud(dataset_dir, url)

    logger.info("Creating and processing YOLO dataset...")
    # process_dataset(dataset_dir, yolo_dataset)
    url_yolo = "https://drive.google.com/uc?export=download&id=1_uq4c2xmZyOpWX9tTrN6fRrB5MElf-q2"
    download_dataset_from_cloud(yolo_dataset, url_yolo, extract_folder=False)
    logger.info("YOLO dataset processing finished.")

def process_dataset(dataset_dir: str, yolo_dataset: str) -> None:
    """
    Process the dataset to create the YOLO dataset

    Args:
        dataset_dir (str): The path to the dataset directory
        yolo_dataset (str): The path to the YOLO dataset

    Returns:
        None    
    """

    if os.path.exists(yolo_dataset):
        return
    
    # Define the path where the nnUNet dataset will be stored
    dataset_path = f"{dataset_dir}/train"
    
    # Create necessary directories for the nnUNet dataset
    _create_yolo_dataset(yolo_dataset) 

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
            
            # Split the dataset
            fold_assign = split_assign(pd)

            # Load the NIFTI files for the current patient and timepoint
            mask = load_nifti_image(f"{td_path}/P{pd}_T{td}_MASK.nii")
            flair = load_nifti_image(f"{td_path}/P{pd}_T{td}_FLAIR.nii")
            t1 = load_nifti_image(f"{td_path}/P{pd}_T{td}_T1.nii")
            t2 = load_nifti_image(f"{td_path}/P{pd}_T{td}_T2.nii")

            data = [flair, t1, t2]

            # Make and save the slices
            _make_save_slices(data, mask, yolo_dataset, fold_assign, pd, td)

def _create_yolo_dataset(yolo_dataset: str) -> None:
    """
    Create the necessary directories for the YOLO dataset

    Args:
        yolo_dataset (str): The path to the YOLO dataset

    Returns:
        None
    """

    # Create the necessary directories for the YOLO dataset
    for i in range(1, 6):
        os.makedirs(f"{yolo_dataset}/fold{i}/images", exist_ok=True)
        os.makedirs(f"{yolo_dataset}/fold{i}/labels", exist_ok=True)

    os.makedirs(f"{yolo_dataset}/Test/images", exist_ok=True)
    os.makedirs(f"{yolo_dataset}/Test/labels", exist_ok=True)

def _make_save_slices(data: list, mask: np.ndarray, yolo_dataset: str, fold_assign: str, pd: int, td: int) -> None:
    """
    Function to create and save the slices of the MRI images and masks

    Args:
        data (list): List of MRI images (FLAIR, T1, T2)
        mask (np.ndarray): The mask of the MRI image
        yolo_dataset (str): The path to the YOLO dataset
        fold_assign (str): The fold assignment for the current patient
        pd (int): The patient ID
        td (int): The timepoint ID

    Returns:
        None
    """

    # Iterate over all sagittal slices in the 3D mask array
    for i in range(mask.shape[0]):
        # Extract the sagittal slice of the mask and extract the contours
        sag_mask = mask[i, :, :]
        sag_mask_ann = extract_contours_mask(sag_mask)

        # Extract corresponding sagittal slices from the FLAIR, T1, and T2 data arrays
        sag_flair = data[0][i, :, :]
        sag_t1 = data[1][i, :, :]
        sag_t2 = data[2][i, :, :]
        sag_mask_ann = extract_contours_mask(sag_mask)

        # Save the image slices as PNG files in the corresponding directory 
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_FLAIR_sagittal_{i}.png", sag_flair, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_T1_sagittal_{i}.png", sag_t1, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_T2_sagittal_{i}.png", sag_t2, cmap="gray")

        # Save the coordinates to the text files
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_FLAIR_sagittal_{i}.txt", 'w') as f:
            f.write(sag_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_T1_sagittal_{i}.txt", 'w') as f:
            f.write(sag_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_T2_sagittal_{i}.txt", 'w') as f:
            f.write(sag_mask_ann)

    # Iterate over all coronal slices in the 3D mask array
    for j in range(mask.shape[1]):
        # Extract the coronal slice of the mask and extract the contours
        cor_mask = mask[:, j, :]
        cor_mask_ann = extract_contours_mask(cor_mask)

        # Extract corresponding coronal slices from the FLAIR, T1, and T2 data arrays
        cor_flair = data[0][:, j, :]
        cor_t1 = data[1][:, j, :]
        cor_t2 = data[2][:, j, :]
    
        # Save the image slices as PNG files in the corresponding directory
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_FLAIR_coronal_{j}.png", cor_flair, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_T1_coronal_{j}.png", cor_t1, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_T2_coronal_{j}.png", cor_t2, cmap="gray")

        # Save the coordinates to the text files
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_FLAIR_coronal_{j}.txt", 'w') as f:
            f.write(cor_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_T1_coronal_{j}.txt", 'w') as f:
            f.write(cor_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_T2_coronal_{j}.txt", 'w') as f:
            f.write(cor_mask_ann)

    # Iterate over all axial slices in the 3D mask array
    for k in range(mask.shape[2]):
        # Extract the axial slice of the mask and extract the contours
        axi_mask = mask[:, :, k]
        axi_mask_ann = extract_contours_mask(axi_mask)

        # Extract corresponding axial slices from the FLAIR, T1, and T2 data arrays
        axi_flair = data[0][:, :, k]
        axi_t1 = data[1][:, :, k]
        axi_t2 = data[2][:, :, k]
        
        # Save the image slices as PNG files in the corresponding directory
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_FLAIR_axial_{k}.png", axi_flair, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_T1_axial_{k}.png", axi_t1, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/images/P{pd}_T{td}_T2_axial_{k}.png", axi_t2, cmap="gray")

        # Save the coordinates to the text files
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_FLAIR_axial_{k}.txt", 'w') as f:
            f.write(axi_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_T1_axial_{k}.txt", 'w') as f:
            f.write(axi_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/labels/P{pd}_T{td}_T2_axial_{k}.txt", 'w') as f:
            f.write(axi_mask_ann)

if __name__ == "__main__":
    yolo_init()
