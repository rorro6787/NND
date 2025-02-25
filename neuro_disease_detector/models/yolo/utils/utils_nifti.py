import nibabel as nib
import numpy as np
import cv2
import torch

def load_nifti_image(file_path: str) -> np.ndarray:
    """Loads a NIfTI (.nii) file and returns its 3D image data as a NumPy array."""

    return nib.load(file_path).get_fdata()

def load_nifti_image_bgr(file_path: str) -> np.ndarray:
    """Loads a NIfTI (.nii) file and returns its 3D image data as a BGR NumPy array."""

    # Obtain image data as a 3D numpy array
    volume = load_nifti_image(file_path)

    # Normalize the volume to the range [0, 255]
    volume_uint8 = volume.astype(np.uint8)

    # Convert the volume to a BGR image
    return np.stack([volume_uint8]*3, axis=-1)

def extract_contours_mask(mask: np.ndarray) -> str:
    """Extracts contours from a binary mask image and returns annotations in YOLO segmentation format."""

    # Convert mask to uint8 for proper contour extraction
    mask = mask.astype(np.uint8)

    # Find contours in the binary mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the dimensions of the mask image
    mask_height, mask_width = mask.shape
    annotations = ""

    # Iterate over each contour found in the mask
    for contour in contours:
        if len(contour) < 3 :
            continue
        # Add the points of the contour (normalized)
        for point in contour:
            px, py = point[0]
            annotations += f"0 {px/mask_width} {py/mask_height}"
        annotations += "\n"
    return annotations
