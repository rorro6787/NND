import os
import cv2
import re

class ImageScan:
    """
    Class to handle image scans for specific patients and slices.

    This class is designed to manage image scan data by parsing the
    filename to extract relevant information such as the patient ID,
    timespan, modality, type of slice, and slice number. It is 
    particularly useful in medical imaging applications where images
    are stored with structured filenames.

    Attributes:
        path (str): The file path to the image scan.
        patient (str): The ID of the patient associated with the scan.
        timespan (str): The timespan or date when the scan was taken.
        modality (str): The imaging modality (e.g., MRI, CT) used for the scan.
        type_slice (str): The type of slice (e.g., axial, sagittal) in the scan.
        number_slice (str): The number of the specific slice in the scan.

    Parameters:
        path (str): The file path to the image scan.
        image_name (str): The name of the image file, expected to be
                          formatted as "patientID_timespan_modality_typeSlice_numberSlice".

    Example:
        image_scan = ImageScan("/images/", "P123_2024-01-01_MRI_axial_001.dcm")
        print(image_scan.patient)  # Output: P123
        print(image_scan.timespan)  # Output: 2024-01-01
    """

    def __init__(self, path: str, image_name: str):
        
        values = re.split(r'[_\.]', image_name)

        self.path = path
        self.patient = values[0]
        self.timespan = values[1]
        self.modality = values[2]
        self.type_slice = values[3]
        self.number_slice = values[4]

    def get_image_name(self):
        """
        Generate the image file name based on the patient's ID, timespan, modality, and slice type.
        
        :return: Image file name (str)
        """
        return f"{self.patient}_{self.timespan}_{self.modality}_{self.type_slice}_{self.number_slice}"
    
    def get_image_path(self, path):
        """
        Generate the full file path for the image.
        
        :param path: Base directory path (str)
        :return: Full path to the image (str)
        """
        return os.path.join(path, f"{self.get_image_name()}.png")
    
    def get_mask_path(self, path):
        """
        Generate the full file path for the mask corresponding to the image.
        
        :param path: Base directory path (str)
        :return: Full path to the mask file (str)
        """
        return os.path.join(path, "MSLesSeg-Dataset-a", "masks", f"{self.patient}_{self.timespan}_{self.type_slice}_{self.number_slice}.png")
    
    def obtain_image_mask(self, path):
        """
        Load the mask file, extract the specific slice, and return it.
        
        :param path: Base directory path (str)
        :return: The extracted mask slice (numpy array)
        """
        return cv2.imread(self.get_mask_path(path), cv2.IMREAD_GRAYSCALE)
