

import subprocess, shutil, json, os, gdown, zipfile
from enum import Enum
FOLD_TO_PATIENT = { "fold1": (1, 7), "fold2": (7, 14), "fold3": (14, 24), "fold4": (24, 41), "fold5": (41, 54) }
TIMEPOINTS_PATIENT = [3,4,4,3,2,3,2,2,3,2,2,2,4,4,1,1,1,1,4,3,1,1,2,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2]                      

def get_timepoints_patient(pd: int) -> int:
    """Returns the timepoints for a given patient."""
    return TIMEPOINTS_PATIENT[pd-1]

def get_patient_by_test_id(test_id: int | str) -> str:
    """ Given a test ID and a list with the number of tests per patient, return the patient to which the test belongs."""
    
    # Convert the test ID to an integer
    test_id = int(test_id)
    current_id = 0

    # Iterate over the number of tests per patient
    for i, num_tests in enumerate(TIMEPOINTS_PATIENT):
        current_id += num_tests
        if test_id <= current_id:
            return i + 1
    # If the test ID is not found, return -1
    return -1

def get_patients_split(split: str) -> tuple:
    """Returns the list of patients for a given split (e.g., train, test)."""
    return FOLD_TO_PATIENT[split]

def split_assign(pd: int) -> str:
    """Assign a patient to a fold based on the patient ID."""

    if pd <= 0 or pd >= 54:
        raise ValueError(f"Invalid patient ID: {pd}")
    
    # Define the boundaries for each fold
    folds = [FOLD_TO_PATIENT[f"fold{i}"][0] for i in range(1, 6)] 

    # Assign the patient to a fold based on their ID
    for i, start in enumerate(folds[:-1]):
        # If the patient ID is within the range of the current fold, return the fold
        if pd >= start and pd < folds[i + 1]:
            return f"fold{i + 1}"
    # If the patient ID is not within the range of any fold, return "test"
    return "fold5"
def download_dataset_from_cloud(folder_name: str, url: str, extract_folder: bool = True) -> None:
    """Downloads and extracts a dataset from a cloud storage URL."""

    if os.path.exists(folder_name):
        return
    
    # Name of the ZIP file to save locally
    dataset_zip = f"{folder_name}.zip"

    # Download the dataset from the cloud storage URL
    gdown.download(url, dataset_zip, quiet=False)
    
    # Extract the dataset from the ZIP file
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        if extract_folder:
            zip_ref.extractall(folder_name)
        else:
            zip_ref.extractall()

    # Remove the ZIP file after extraction
    os.remove(dataset_zip)
class Trainer(Enum):
    EPOCHS_1 = "nnUNetTrainer_1epoch"
    EPOCHS_5 = "nnUNetTrainer_5epochs"
    EPOCHS_10 = "nnUNetTrainer_10epochs"
    EPOCHS_20 = "nnUNetTrainer_20epochs"
    EPOCHS_50 = "nnUNetTrainer_50epochs"
    EPOCHS_100 = "nnUNetTrainer_100epochs"
    EPOCHS_250 = "nnUNetTrainer_250epochs"
    EPOCHS_500 = "nnUNetTrainer_500epochs"
    EPOCHS_750 = "nnUNetTrainer_750epochs"
    EPOCHS_2000 = "nnUNetTrainer_2000epochs"
    EPOCHS_4000 = "nnUNetTrainer_4000epochs"
    EPOCHS_8000 = "nnUNetTrainer_8000epochs"

class Fold(Enum):
    FOLD_1 = "0"
    FOLD_2 = "1"
    FOLD_3 = "2"
    FOLD_4 = "3"
    FOLD_5 = "4"

class Configuration(Enum):
    SIMPLE_2D = "2d"
    FULL_3D = "3d_fullres"

cwd = os.getcwd()

class nnUNet:
    """
    Class representing the nnUNet model for training, validation, and testing.

    Attributes:
        dataset_id (str):
            The ID of the dataset.

        configuration (Configuration):
            The configuration object specifying model settings.

        fold (Fold): 
            The fold used for cross-validation.

        trainer (Trainer): 
            The trainer object used for training the model.

        k (int):
            Number of folds for cross-validation.

        nnunet_cwd (str): 
            The base directory for nnUNet.

        og_dataset (str): 
            Path to the original dataset.

        dataset_name (str): 
            The name of the dataset.

        nnUNet_datapath (str): 
            The path where nnUNet data is stored.

        nnUNet_raw (str): 
            Path to raw nnUNet data.

        nnUNet_preprocessed (str): 
            Path to preprocessed nnUNet data.

        nnUNet_results (str): 
            Path to the results of the nnUNet.

        train_results (str): 
            The directory where training results are stored.

        val_results (str): 
            The directory for validation results.

        test_results (str): 
            The directory for testing results.

        logger (Logger):
            The logger object for the nnUNet model.

    Methods:
        __init__(self, dataset_id: str, configuration: Configuration, fold: Fold, trainer: Trainer):
            Initializes the nnUNet class with dataset, configuration, fold, and trainer.
        
        init(self, csv_path: str):
            Initializes the nnUNet environment, processes the dataset, trains the model,
            and writes results to the specified CSV file.

        write_results(self, csv_path: str):
            Writes the results of the nnUNet model to a CSV file.

        evaluate_test_results(self):
            Evaluates the test results using the trained nnUNet model.
        
        inference_test(self):
            Performs inference on the test data using the trained nnUNet model.

        train_nnUNet(self):
            Trains the nnUNet model using the specified configuration, fold, and trainer.
        
        process_dataset(self):
            Processes the dataset using nnUNet.
        
        configure_environment(self):
            Configures the environment variables for nnUNet.

        create_nnu_dataset(self):
            Creates the nnUNet dataset from the MSLesSeg-Dataset.
    """

    def __init__(self, dataset_id: str, configuration: Configuration, fold: Fold, trainer: Trainer):
        """
        Initializes the nnUNet model with necessary parameters and file paths.

        Args:
            dataset_id (str): 
                The ID of the dataset.

            configuration (Configuration):
                The configuration settings for the model.

            fold (Fold): 
                The fold for cross-validation.

            trainer (Trainer): 
                The trainer object used to train the model.

        Example:
            >>> from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
            >>> from neuro_disease_detector.models.nnUNet.nnUNet_pipeline import nnUNet
            >>> 
            >>> dataset_id = "024"
            >>> configuration = Configuration.FULL_3D
            >>> fold = Fold.FOLD_5
            >>> trainer = Trainer.EPOCHS_100
            >>> nnUNet = nnUNet(dataset_id, configuration, fold, trainer)
        """

        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.trainer = trainer
        self.k = 5

        self.nnunet_cwd = f"{cwd}/nnu_net"
        self.og_dataset = f"{cwd}/MSLesSeg-Dataset"
        self.dataset_name = f"Dataset{dataset_id}_MSLesSeg"
        self.nnUNet_datapath = f"{self.nnunet_cwd}/nnUNet_raw/{self.dataset_name}"

        self.nnUNet_raw = f"{self.nnunet_cwd}/nnUNet_raw"
        self.nnUNet_preprocessed = f"{self.nnunet_cwd}/nnUNet_preprocessed"
        self.nnUNet_results = f"{self.nnunet_cwd}/nnUNet_results"

        self.nnUNet_datapath = f"{self.nnUNet_raw}/{self.dataset_name}"

        self.train_results = f"{self.nnUNet_results}/Dataset{dataset_id}_MSLesSeg/{trainer.value}__nnUNetPlans__{configuration.value}/fold_{fold.value}"
        self.val_results = f"{self.train_results}/validation"
        self.test_results = f"{self.train_results}/test"


    

    def create_nnu_dataset(self) -> None:
        """
        Create the nnUNet dataset from the MSLesSeg-Dataset.

        Args:
            self.nnUNet_datapath (str): 
                The path to the nnUNet dataset.

            self.og_dataset (str): 
                The path to the original dataset.

        Returns:
            None

        Example:
            >>> from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
            >>> from neuro_disease_detector.models.nnUNet.nnUNet_pipeline import nnUNet
            >>>
            >>> dataset_id = "024"
            >>> configuration = Configuration.FULL_3D
            >>> fold = Fold.FOLD_5
            >>> trainer = Trainer.EPOCHS_100
            >>> nnUNet = nnUNet(dataset_id, configuration, fold, trainer)
            >>> nnUNet.create_nnu_dataset()
        """

        download_dataset_from_cloud(self.og_dataset, 
                                    "https://drive.google.com/uc?export=download&id=1TM4ciSeiyl-ri4_Jn4-aMOTDSSSHM6XB", 
                                    extract_folder=False
        )

        self.logger.info("Creating nnUNet dataset...")
        if os.path.exists(self.nnUNet_datapath):
            return
        
        # Define the path where the nnUNet dataset will be stored
        dataset_path = f"{self.og_dataset}/train"
        
        # Create necessary directories for the nnUNet dataset
        os.makedirs(self.nnUNet_datapath)
        os.makedirs(f"{self.nnUNet_datapath}/imagesTr")  
        os.makedirs(f"{self.nnUNet_datapath}/imagesTs")  
        os.makedirs(f"{self.nnUNet_datapath}/labelsTr")  
        os.makedirs(f"{self.nnUNet_datapath}/labelsTs")  
        
        # Initialize a unique id counter for each subject
        id = 0

        # Iterate over the subjects in the dataset
        for pd in range(1, 54):
            # Skip the subject with id 30
            if pd == 30:
                continue

            # Get the number of timepoints available for this patient.
            num_tp = get_timepoints_patient(pd)
            pd_path = f"{dataset_path}/P{pd}"

            # Iterate over each timepoint for the current patient.
            for td in range(1, num_tp+1):
                # Define the path for the timepoint folder
                td_path = f"{pd_path}/T{td}"
                id += 1
                
                # Define the paths to the image and mask files
                flair_path = f"{td_path}/P{pd}_T{td}_FLAIR.nii"
                t1_path = f"{td_path}/P{pd}_T{td}_T1.nii"
                t2_path = f"{td_path}/P{pd}_T{td}_T2.nii"
                mask_path = f"{td_path}/P{pd}_T{td}_MASK.nii"

                # Assign the patient to a fold based on their ID
                train_test = "Ts" if split_assign(pd) == "Test" else "Tr"  

                # Copy the images and mask to the appropriate directories
                shutil.copy(flair_path, f"{self.nnUNet_datapath}/images{train_test}/BRATS_{id}_0000.nii.gz")
                shutil.copy(t1_path, f"{self.nnUNet_datapath}/images{train_test}/BRATS_{id}_0001.nii.gz")
                shutil.copy(t2_path, f"{self.nnUNet_datapath}/images{train_test}/BRATS_{id}_0002.nii.gz")
                shutil.copy(mask_path, f"{self.nnUNet_datapath}/labels{train_test}/BRATS_{id}.nii.gz")

        # Copy the dataset.json file to the nnUNet dataset directory
        #shutil.copy(CONFIG_NNUNET, f"{self.nnUNet_datapath}/dataset.json")
    
def _remove_files(folder_path: str):
    """Removes all files in the folder that start with 'BRATS_'."""
    for file in os.listdir(folder_path):
        if file.startswith("BRATS_"):
            os.remove(f"{folder_path}/{file}")
        


if __name__ == "__main__":
    import shutil
    nnunet = nnUNet("023", Configuration.FULL_3D, Fold.FOLD_5, Trainer.EPOCHS_100)
    nnunet.create_nnu_dataset()
    #nnunet.configure_environment()
    #nnunet.process_dataset()

    # shutil.make_archive('nnu_net', 'zip', '/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/models/nnUNet/nnu_net')
