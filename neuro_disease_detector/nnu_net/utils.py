import matplotlib.pyplot as plt
from vedo import Volume, show
import nibabel as nib
import numpy as np
import os

cwd = os.getcwd()

timepoints_patient = [3,4,4,3,2,3,   
                      2,2,3,2,2,4,2,  
                      4,1,1,1,1,4,3,1,1,
                      2,1,1,1,1,2,1,0,2,1,2,1,1,1,   
                      1,1,1,1,1,1,1,2,2,2,2,1,1,     
                      1,1,1,2]

def _get_patient_by_test_id(test_id: int | str):
    """ 
    Given a test ID and a list with the number of tests per patient,
    return the patient to which the test belongs.

    Args:
        test_id (int | str): The ID of the test.

    Returns:
        str: The patient to which the test belongs.
    """

    test_id = int(test_id)
    current_id = 0

    for i, num_tests in enumerate(timepoints_patient):
        current_id += num_tests
        if test_id <= current_id:
            return f"P{i + 1}"
        
    return "ID not found"

def show_volumes(vol: str, mask: str, prediction: str):
    """
    Show the volumes in a 3D plot

    Args:
        vol: The volume to show
        mask: The mask volume to show
        prediction: The model's prediction volume
    
    Returns:
        None
    """
    
    vol = Volume(vol)
    mask = Volume(mask).cmap("Reds").add_scalarbar("Ground Truth")
    prediction = Volume(prediction).cmap("Greens").add_scalarbar("Model Prediction", pos=(0.1, 0.06))

    show(vol, mask, prediction, axes=1)

def _display_slices(slices: list):
    """
    Function to plot a slice of an MRI image

    Args:
        slices: List of 2D slices to plot

    Returns:
        None
    """
    # Normalize slices to [0, 1] for proper blending
    slices = [
        np.clip(slice_data / np.max(slice_data), 0, 1) if np.max(slice_data) > 0 else slice_data
        for slice_data in slices
    ]

    # Separate the images
    original = slices[0]
    ground_truth = slices[1]
    prediction = slices[2]

    # Create a new RGB image for overlay
    overlay = np.zeros((*original.shape, 3))  
    overlay[..., 0] = prediction  
    overlay[..., 2] = ground_truth 

    overlay_truth = np.zeros((*original.shape, 3)) 
    overlay_truth[..., 2] = ground_truth 

    overlay_pred = np.zeros((*original.shape, 3))  
    overlay_pred[..., 0] = prediction  

    # Plot the two images side by side
    _, axes = plt.subplots(1, 4, figsize=(20, 20))

    # Left axis: Original grayscale image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original MRI Slice")
    axes[0].axis('off')

    # Right axis: Superimposed image
    axes[1].imshow(original, cmap='gray')  
    axes[1].imshow(overlay_truth, alpha=0.5)  
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # Right axis: Superimposed image
    axes[2].imshow(original, cmap='gray')  
    axes[2].imshow(overlay_pred, alpha=0.5)  
    axes[2].set_title("Model's Prediction")
    axes[2].axis('off')

    # Right axis: Superimposed image
    axes[3].imshow(original, cmap='gray')  
    axes[3].imshow(overlay, alpha=0.5)  
    axes[3].set_title("Overlap")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

def display_slices(slice_type: str, slice_index: int, nifti_files: list):
    """
    Function to display a slice of an MRI image

    Args:
        slice_type: Type of slice to display (Axial, Coronal, Sagittal)
        slice_index: Index of the slice to display
    
    Returns:
        None
    """

    os.rename(f"{nifti_files[0]}.gz", nifti_files[0])
    os.rename(f"{nifti_files[1]}.gz", nifti_files[1])

    # Load NIFTI files and extract the slice
    slices = []
    for file in nifti_files:
        nii_img = nib.load(file)  
        data = nii_img.get_fdata()  
        if slice_type == "Axial":
            slice_data = data[:, :, slice_index]
        elif slice_type == "Coronal":
            slice_data = data[:, slice_index, :]
        elif slice_type == "Sagittal":
            slice_data = data[slice_index, :, :]

        slices.append(slice_data)

    os.rename(nifti_files[0], f"{nifti_files[0]}.gz")
    os.rename(nifti_files[1], f"{nifti_files[1]}.gz")

    # Plot the slices
    _display_slices(slices)

if __name__ == "__main__":
    nnUNet_raw = f"{cwd}/nnUNet_raw/Dataset024_MSLesSeg"
    nnUNet_prediction = f"{nnUNet_raw}/nnUNet_tests_0/BRATS_89.nii.gz"
    mask = f"{nnUNet_raw}/labelsTs/BRATS_89.nii.gz"
    flair = f"{nnUNet_raw}/imagesTs/BRATS_89_0000.nii.gz"
    show_volumes(flair, mask, nnUNet_prediction)
