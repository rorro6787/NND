import matplotlib.pyplot as plt
import numpy as np
import os 

from neuro_disease_detector.models.yolo.utils.utils_nifti import extract_contours_mask, load_nifti_image
from neuro_disease_detector.utils.utils_dataset import get_timepoints_patient
from neuro_disease_detector.utils.utils_dataset import split_assign

cwd = os.getcwd()

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

        # Get the number of timepoints available for this patient.
        num_tp = get_timepoints_patient(pd)
        pd_path = f"{dataset_path}/P{pd}"

        # Iterate over each timepoint for the current patient.
        for tp in range(1, num_tp+1):
            # Define the path for the timepoint folder
            tp_path = f"{pd_path}/T{tp}"
            
            # Split the dataset
            fold_assign = split_assign(pd)

            # Load the NIFTI files for the current patient and timepoint
            mask = load_nifti_image(f"{tp_path}/P{pd}_T{tp}_MASK.nii")
            flair = load_nifti_image(f"{tp_path}/P{pd}_T{tp}_FLAIR.nii")
            t1 = load_nifti_image(f"{tp_path}/P{pd}_T{tp}_T1.nii")
            t2 = load_nifti_image(f"{tp_path}/P{pd}_T{tp}_T2.nii")

            data = [flair, t1, t2]

            # Make and save the slices
            _make_save_slices(data, mask, yolo_dataset, fold_assign, pd, tp)

def _create_yolo_dataset(yolo_dataset: str) -> None:
    """
    Create the necessary directories for the YOLO dataset

    Args:
        yolo_dataset (str): The path to the YOLO dataset

    Returns:
        None
    """

    # Create the necessary directories for the YOLO dataset
    planes = ["sagittal", "coronal", "axial"]
    for plane in planes:
        for i in range(1, 6):
            os.makedirs(f"{yolo_dataset}/fold{i}/{plane}/images", exist_ok=True)
            os.makedirs(f"{yolo_dataset}/fold{i}/{plane}/labels", exist_ok=True)

        os.makedirs(f"{yolo_dataset}/Test/{plane}/images", exist_ok=True)
        os.makedirs(f"{yolo_dataset}/Test/{plane}/labels", exist_ok=True)

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

        # Save the image slices as PNG files in the corresponding directory 
        plt.imsave(f"{yolo_dataset}/{fold_assign}/sagittal/images/P{pd}_T{td}_FLAIR_sagittal_{i}.png", sag_flair, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/sagittal/images/P{pd}_T{td}_T1_sagittal_{i}.png", sag_t1, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/sagittal/images/P{pd}_T{td}_T2_sagittal_{i}.png", sag_t2, cmap="gray")

        # Save the coordinates to the text files
        with open(f"{yolo_dataset}/{fold_assign}/sagittal/labels/P{pd}_T{td}_FLAIR_sagittal_{i}.txt", 'w') as f:
            f.write(sag_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/sagittal/labels/P{pd}_T{td}_T1_sagittal_{i}.txt", 'w') as f:
            f.write(sag_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/sagittal/labels/P{pd}_T{td}_T2_sagittal_{i}.txt", 'w') as f:
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
        plt.imsave(f"{yolo_dataset}/{fold_assign}/coronal/images/P{pd}_T{td}_FLAIR_coronal_{j}.png", cor_flair, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/coronal/images/P{pd}_T{td}_T1_coronal_{j}.png", cor_t1, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/coronal/images/P{pd}_T{td}_T2_coronal_{j}.png", cor_t2, cmap="gray")

        # Save the coordinates to the text files
        with open(f"{yolo_dataset}/{fold_assign}/coronal/labels/P{pd}_T{td}_FLAIR_coronal_{j}.txt", 'w') as f:
            f.write(cor_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/coronal/labels/P{pd}_T{td}_T1_coronal_{j}.txt", 'w') as f:
            f.write(cor_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/coronal/labels/P{pd}_T{td}_T2_coronal_{j}.txt", 'w') as f:
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
        plt.imsave(f"{yolo_dataset}/{fold_assign}/axial/images/P{pd}_T{td}_FLAIR_axial_{k}.png", axi_flair, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/axial/images/P{pd}_T{td}_T1_axial_{k}.png", axi_t1, cmap="gray")
        plt.imsave(f"{yolo_dataset}/{fold_assign}/axial/images/P{pd}_T{td}_T2_axial_{k}.png", axi_t2, cmap="gray")

        # Save the coordinates to the text files
        with open(f"{yolo_dataset}/{fold_assign}/axial/labels/P{pd}_T{td}_FLAIR_axial_{k}.txt", 'w') as f:
            f.write(axi_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/axial/labels/P{pd}_T{td}_T1_axial_{k}.txt", 'w') as f:
            f.write(axi_mask_ann)
        with open(f"{yolo_dataset}/{fold_assign}/axial/labels/P{pd}_T{td}_T2_axial_{k}.txt", 'w') as f:
            f.write(axi_mask_ann)
