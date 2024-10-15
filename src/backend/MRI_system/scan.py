# Class to represent a scan of a medical image.
class Scan:
    def __init__(self, patient: str=None, timespan: str=None, modality: str=None, type_slice: str=None):
        """
        Initialize the Scan object with the patient's ID, timespan, modality, and slice type.
        
        :param patient: ID of the patient (str)
        :param timespan: Time of the scan (str)
        :param modality: Modality of the scan, e.g., MRI, CT (str)
        :param type_slice: Type of the slice (axial, sagittal, coronal) (str)
        """
        self.patient = patient
        self.timespan = timespan
        self.modality = modality
        self.type_slice = type_slice

    def set_patient(self, patient):
        """
        Set the patient ID.
        :param patient: ID of the patient (str)
        """
        self.patient = patient
    
    def set_timespan(self, timespan):
        """
        Set the scan timespan.
        :param timespan: Time of the scan (str)
        """
        self.timespan = timespan
    
    def set_modality(self, modality):
        """
        Set the modality of the scan.
        :param modality: Modality of the scan (str)
        """
        self.modality = modality
    
    def set_type_slice(self, type_slice):
        """
        Set the type of slice.
        :param type_slice: Type of the slice (axial, sagittal, coronal) (str)
        """
        self.type_slice = type_slice


import os
import nibabel as nib

# Class to handle image scans for specific patients and slices.
class ImageScan:
    def __init__(self, patient: str, timespan: str, number_slice: str, modality: int, type_slice: int):
        """
        Initialize the ImageScan object with patient ID, scan timespan, slice number, modality, and slice type.
        
        :param patient: ID of the patient (str)
        :param timespan: Time of the scan (str)
        :param number_slice: Slice number of the scan (str)
        :param modality: Modality of the scan (int) -> {0: "FLAIR", 1: "T1", 2: "T2"}
        :param type_slice: Type of the slice (int) -> {0: "axial", 1: "sagittal", 2: "coronal"}
        """
        self.patient = f"P{patient}"
        self.timespan = f"T{timespan}"
        self.number_slice = number_slice

        # Dictionaries to map the integer values to modality and slice type.
        dic_modal = {0: "FLAIR", 1: "T1", 2: "T2"}
        dic_type = {0: "axial", 1: "sagittal", 2: "coronal"}
        self.modality = dic_modal[modality]
        self.type_slice = dic_type[type_slice]

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
        return os.path.join(path, "MSLesSeg-Dataset", "train", self.patient, self.timespan, f"{self.patient}_{self.timespan}_MASK.nii")
    
    def obtain_image_mask(self, path):
        """
        Load the mask file, extract the specific slice, and return it.
        
        :param path: Base directory path (str)
        :return: The extracted mask slice (numpy array)
        """
        img_mask = nib.load(self.get_mask_path(path))  # Load the mask file
        img_mask = img_mask.get_fdata()  # Convert mask data to numpy array
        slice_mask = None
        
        # Extract the slice based on the type (axial, sagittal, coronal).
        if self.type_slice == "axial":
            slice_mask = img_mask[:, :, int(self.number_slice)]
        elif self.type_slice == "sagittal":
            slice_mask = img_mask[int(self.number_slice), :, :]
        elif self.type_slice == "coronal":
            slice_mask = img_mask[:, int(self.number_slice), :]

        return slice_mask
