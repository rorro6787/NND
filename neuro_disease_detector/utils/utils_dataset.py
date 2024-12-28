import nibabel as nib
import numpy as np
import cv2

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

def load_nifti_image_bgr(file_path: str) -> np.ndarray:
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
    volume = img.get_fdata()

    # Normalize the volume to the range [0, 255]
    volume_uint8 = volume.astype(np.uint8)

    # Add a BGR channel to the 3D volume
    volume_bgr = np.stack([volume_uint8]*3, axis=-1)

    return volume_bgr

def extract_contours_mask(mask: np.ndarray) -> str:
    """
    Extracts the normalized coordinates of contours from a binary mask image.

    Parameters:
    - mask: np.ndarray
        A 2D NumPy array representing the black and white image, where white pixels
        have a value of 255 and black pixels have a value of 0.

    Returns:
    - str:
        A string containing the normalized coordinates of contours found in the image in the format:
        "0 <x_center> <y_center> <width> <height> <x1> <y1> <x2> <y2> ... <xn> <yn>"
        for each contour. Each coordinate is normalized to fall between 0 and 1,
        and the values are formatted to six decimal places.
    """

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