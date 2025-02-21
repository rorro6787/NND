
from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud
from neuro_disease_detector.utils.utils_dataset import get_timepoints_patient
from neuro_disease_detector.utils.utils_dataset import split_assign
from neuro_disease_detector.__init__ import CONFIG_NNUNET_SPLIT
from neuro_disease_detector.__init__ import CONFIG_NNUNET
from neuro_disease_detector.logger import get_logger
from neuro_disease_detector.utils.utils_dataset import write_results_csv

import subprocess, shutil, os

logger = get_logger(__name__)
cwd = os.getcwd()

class nnUNet:
    def __init__(self, dataset_id: str, configuration: Configuration, fold: Fold, trainer: Trainer):
        self.dataset_id = dataset_id
        self.configuration = configuration
        self.fold = fold
        self.trainer = trainer

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

    def init(self, csv_path: str):
        download_dataset_from_cloud(self.og_dataset, "https://drive.google.com/uc?export=download&id=1A3ZpXHe-bLpaAI7BjPTSkZHyQwEP3pIi")

        self.create_nnu_dataset()  
        self.configure_environment()   
        self.process_dataset()

        _remove_files(f"{self.nnUNet_datapath}/imagesTr")
        _remove_files(f"{self.nnUNet_datapath}/labelsTr") 

        self.train_nnUNet()

        _remove_files(self.val_results) 

        self.inference_test()     
        self.evaluate_test_results()

        model_type = "nnUNet3D" if self.configuration == Configuration.FULL_3D else "nnUNet2D"

        write_results_csv(csv_path, f"{self.val_results}/summary.json", model_type, self.dataset_id, self.fold.value, "val")
        write_results_csv(csv_path, f"{self.test_results}/summary.json", model_type, self.dataset_id, self.fold.value, "test")

        _remove_files(self.test_results)

    def evaluate_test_results(self) -> None:
        """
        Evaluate the test results using the trained nnUNet model.

        Args:
            nnUNet_datapath (str): The path to the nnUNet dataset.
            test_results (str): The path to the test results.

        Returns:
            None
        """

        logger.info("Evaluating test results...")
        test_path = f"{self.nnUNet_datapath}/labelsTs"

        # Define the command to evaluate the test results
        command = ["nnUNetv2_evaluate_folder", 
                test_path,
                self.test_results, 
                "-djfile", f"{self.test_results}/dataset.json", 
                "-pfile", f"{self.test_results}/plans.json"]
        
        subprocess.run(command, env=os.environ, check=True)
        logger.info("nnUNet pipeline completed.")

    def inference_test(self) -> None:
        """
        Perform inference on the test data using the trained nnUNet model.

        Args:
            nnUNet_datapath (str): The path to the nnUNet dataset.
            test_results (str): The path to save the output predictions.
            dataset_id (str): The ID of the dataset.
            configuration (Configuration): The configuration used for training.
            trainer (Trainer): The trainer used for training.
            fold (Fold): The fold used for training.

        Returns:
            None
        """

        # Create the necessary directories for the output predictions
        logger.info("Performing inference on test data...")
        os.makedirs(self.test_results, exist_ok=True)
        test_path = f"{self.nnUNet_datapath}/imagesTs"

        # Define the command to perform inference on the test data
        command = ["nnUNetv2_predict", 
                "-i", test_path, 
                "-o", self.test_results, 
                "-d", self.dataset_id,
                "-c", self.configuration.value, 
                "-tr", self.trainer.value,
                "-f", self.fold.value]
        
        subprocess.run(command, env=os.environ, check=True)

    def train_nnUNet(self) -> None:
        """
        Train the nnUNet model using the specified configuration, fold, and trainer.

        Args:
            dataset_id (str): The ID of the dataset.
            configuration (Configuration): The configuration to use for training.
            fold (Fold): The fold to use for training.
            trainer (Trainer): The trainer to use for training.

        Returns:
            None
        """

        # Define the command to train the nnUNet model
        logger.info(f"Training nnUNet model for fold{self.fold.value}...")
        command = ["nnUNetv2_train", 
                   self.dataset_id, 
                   self.configuration.value, 
                   self.fold.value, 
                   "-tr", self.trainer.value]
        
        subprocess.run(command, env=os.environ, check=True)

    def process_dataset(self) -> None:
        """
        Process the dataset using nnUNet.

        Args:
            process_data (str): The path to the preprocessed data directory.
            dataset_name (str): The name of the dataset.
            dataset_id (str): The ID of the dataset.

        Returns:
            None
        """

        logger.info("Processing dataset...")
        if os.path.exists(f"{self.nnUNet_preprocessed}/{self.dataset_name}"):
            return
        
        # Define and run the command to preprocess the dataset
        command = ["nnUNetv2_plan_and_preprocess", "-d", self.dataset_id, "--verify_dataset_integrity", "-np", "1"]
        subprocess.run(command, env=os.environ, check=True)

        # Copy the splits_final.json file to the appropriate directory
        shutil.copy(CONFIG_NNUNET_SPLIT, f"{self.nnUNet_preprocessed}/{self.dataset_name}/splits_final.json")

    def configure_environment(self) -> None:
        """
        Configure the environment variables for nnUNet.

        Args:
            raw_data (str): The path to the raw data directory.
            process_data (str): The path to the preprocessed data directory.
            results (str): The path to the results directory.

        Returns:
            None
        """

        # Create the necessary directories for nnUNet
        logger.info("Configuring nnUNet environment...")
        os.makedirs(self.nnUNet_raw, exist_ok=True)
        os.makedirs(self.nnUNet_preprocessed, exist_ok=True)
        os.makedirs(self.nnUNet_results, exist_ok=True)

        # Set the environment variables for nnUNet
        os.environ["nnUNet_raw"] = self.nnUNet_raw
        os.environ["nnUNet_preprocessed"] = self.nnUNet_preprocessed
        os.environ["nnUNet_results"] = self.nnUNet_results

    def create_nnu_dataset(self) -> None:
        """
        Create the nnUNet dataset from the MSLesSeg-Dataset.

        Args:
            dataset_dir (str): The path to the MSLesSeg-Dataset.
            nnUNet_datapath (str): The path to the nnUNet dataset.

        Returns:
            None
        """

        logger.info("Creating nnUNet dataset...")
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
        shutil.copy(CONFIG_NNUNET, f"{self.nnUNet_datapath}/dataset.json")
    
def _remove_files(folder_path: str):
    """Removes all files in the folder that start with 'BRATS_'."""
    for file in os.listdir(folder_path):
        if file.startswith("BRATS_"):
            os.remove(f"{folder_path}/{file}")

if __name__ == "__main__":
    dataset_id = "024"
    configuration = Configuration.FULL_3D
    fold = Fold.FOLD_5
    trainer = Trainer.EPOCHS_100
    nnUNet = nnUNet(dataset_id, configuration, fold, trainer)
    nnUNet.init(f"{os.getcwd()}/res.csv")