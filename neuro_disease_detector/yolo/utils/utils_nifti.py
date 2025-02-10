import nibabel as nib
import numpy as np
import cv2
import torch

def load_nifti_image(file_path: str) -> np.ndarray:
    """Loads a NIfTI (.nii) file and returns its 3D image data as a NumPy array."""

    return nib.load(file_path).get_fdata()

def load_nifti_image_tensor(file_path: str) -> np.ndarray:
    """Loads a NIfTI (.nii) file and returns its 3D image data as a PyTorch tensor."""

    # Obtain image data as a 3D numpy array
    volume = nib.load(file_path).get_fdata()

    # Normalize the volume to the range (0.0, 1.0) and add a RGB channel
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    volume_rgb = np.stack([volume]*3, axis=-1)

    # Convert the volume to a PyTorch tensor
    return torch.tensor(volume_rgb)

def load_nifti_image_bgr(file_path: str) -> np.ndarray:
    """Loads a NIfTI (.nii) file and returns its 3D image data as a BGR NumPy array."""

    # Obtain image data as a 3D numpy array
    volume = nib.load(file_path).get_fdata()

    # Normalize the volume to the range [0, 255]
    volume_uint8 = volume.astype(np.uint8)

    # Convert the volume to a BGR image
    return np.stack([volume_uint8]*3, axis=-1)

def extract_contours_mask(mask: np.ndarray) -> str:
    """Extracts contours from a binary mask image and returns annotations in YOLO format."""

    # Convert mask to uint8 to ensure proper format for contour extraction
    mask = mask.astype(np.uint8)

    # Find contours in the binary mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the dimensions of the mask image
    mask_height, mask_width = mask.shape

    # Initialize a string to hold the annotations
    annotations = ""

    # Iterate over each contour found in the mask
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
