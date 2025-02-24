
from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud
from neuro_disease_detector.utils.utils_dataset import get_patient_by_test_id
from neuro_disease_detector.utils.utils_dataset import get_timepoints_patient
from neuro_disease_detector.utils.utils_dataset import write_results_csv
from neuro_disease_detector.utils.utils_dataset import split_assign
from neuro_disease_detector.__init__ import CONFIG_NNUNET_SPLIT
from neuro_disease_detector.__init__ import CONFIG_NNUNET
from neuro_disease_detector.logger import get_logger

import subprocess, shutil, json, os

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

        self.logger = get_logger(__name__)

    def execute_pipeline(self, csv_path: str):
        """
        Initializes the nnUNet environment, processes the dataset, trains the model,
        and writes the results to a CSV file.

        Args:
            csv_path (str): 
                The path to the CSV file where results will be written.

        Steps:
            1. Downloads the dataset from the cloud.
            2. Prepares the dataset by creating necessary files and directories.
            3. Configures the environment and preprocesses the dataset.
            4. Trains the nnUNet model.
            5. Runs inference and evaluates the results.
            6. Writes the results to the specified CSV file.

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
            >>> nnUNet.init("model_study.csv")
        """

        self.create_nnu_dataset()  
        self.configure_environment()   
        self.process_dataset()
        self.train_nnUNet()
        self.inference_test()     
        self.evaluate_test_results()
        self.write_results(csv_path)
        
        _remove_files(f"{self.nnUNet_datapath}/imagesTr")
        _remove_files(f"{self.nnUNet_datapath}/labelsTr") 
        _remove_files(self.val_results) 
        _remove_files(self.test_results)

        if self.configuration == Configuration.FULL_3D and self.fold == Fold.FOLD_5:
            _remove_folder(self.nnUNet_datapath)
            _remove_folder(f"{self.nnUNet_preprocessed}/Dataset{self.dataset_id}_MSLesSeg")

    def write_results(self, csv_path: str) -> None:
        """
        Write the results of the nnUNet model to a CSV file.

        Args:
            self.test_results (str): 
                The path to the test results.

            self.configuration (Configuration): 
                The configuration used for training.

            self.dataset_id (str): 
                The ID of the dataset.

            csv_path (str): 
                The path to the CSV file where results will be written.
        
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
            >>> nnUNet.configure_environment()
            >>> nnUNet.process_dataset()
            >>> nnUNet.train_nnUNet()
            >>> nnUNet.inference_test()
            >>> nnUNet.evaluate_test_results()
            >>> nnUNet.write_results("model_study.csv")
        """
        
        def _write_overall(csv_path: str, results_path: str, algorithm: str, overall: str, execution_id: int):
            """Write the overall results (Val or Test) to the CSV file."""

            with open(results_path, "r") as f:
                # Load the summary.json file
                data = json.load(f)
                foreground_mean = data["foreground_mean"]
                dsc = foreground_mean["Dice"]
                iou = foreground_mean["IoU"]

                # Write the overall results to the CSV file
                write_results_csv(csv_path, algorithm, overall, "DSC", execution_id, dsc)
                write_results_csv(csv_path, algorithm, overall, "IoU", execution_id, iou)

                # Return the data
                return data
                
        # Define the constants for the results
        TEST_RESULTS_PATH = f"{self.test_results}/summary.json"
        VAL_RESULTS_PATH = f"{self.val_results}/summary.json"
        ALGORITHM = "nnUNet3D" if self.configuration == Configuration.FULL_3D else "nnUNet2D"
        EXECUTON_ID = int(self.dataset_id) * self.k + int(self.fold.value) + 1
        BRATS_INITIAL_INDEX = 84

        _ = _write_overall(csv_path, VAL_RESULTS_PATH, ALGORITHM, "GlobalV", EXECUTON_ID)
        data = _write_overall(csv_path, TEST_RESULTS_PATH, ALGORITHM, "GlovalT", EXECUTON_ID)

        brats_i = BRATS_INITIAL_INDEX
        for instance_metrics in data["metric_per_case"]:
            # Write the results for each instance to the CSV file
            instance_metrics = instance_metrics["metrics"]["1"]
            dsc = instance_metrics["Dice"]
            iou = instance_metrics["IoU"]

            # Get the patient and timepoint IDs for the current test instance
            patient_id = get_patient_by_test_id(brats_i)
            timepoint_id = get_timepoints_patient(patient_id)

            # Write the results to the CSV file
            write_results_csv(csv_path, ALGORITHM, f"P{patient_id}T{timepoint_id}", "DSC", EXECUTON_ID, dsc)
            write_results_csv(csv_path, ALGORITHM, f"P{patient_id}T{timepoint_id}", "IoU", EXECUTON_ID, iou)

            brats_i += 1

    def evaluate_test_results(self) -> None:
        """
        Evaluate the test results using the trained nnUNet model.

        Args:
            self.test_results (str): 
                The path to the test results.

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
            >>> nnUNet.configure_environment()
            >>> nnUNet.process_dataset()
            >>> nnUNet.train_nnUNet()
            >>> nnUNet.inference_test()
            >>> nnUNet.evaluate_test_results()
        """

        self.logger.info("Evaluating test results...")
        test_path = f"{self.nnUNet_datapath}/labelsTs"

        # Define the command to evaluate the test results
        command = ["nnUNetv2_evaluate_folder", 
                test_path,
                self.test_results, 
                "-djfile", f"{self.test_results}/dataset.json", 
                "-pfile", f"{self.test_results}/plans.json"]
        
        subprocess.run(command, env=os.environ, check=True)
        self.logger.info("nnUNet pipeline completed.")

    def inference_test(self) -> None:
        """
        Perform inference on the test data using the trained nnUNet model.

        Args:
            self.test_results (str): 
                The path to the test results.

            self.nnUNet_datapath (str): 
                The path to the nnUNet dataset.

            self.dataset_id (str): 
                The ID of the dataset.

            self.configuration (Configuration):
                The configuration used for training.

            self.fold (Fold): 
                The fold used for training.

            self.trainer (Trainer): 
                The trainer used for training.
            
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
            >>> nnUNet.configure_environment()
            >>> nnUNet.process_dataset()
            >>> nnUNet.train_nnUNet()
            >>> nnUNet.inference_test()
        """

        # Create the necessary directories for the output predictions
        self.logger.info("Performing inference on test data...")
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
            self.dataset_id (str): 
                The ID of the dataset.

            self.configuration (Configuration): 
                The configuration used for training.

            self.fold (Fold): 
                The fold used for training.

            self.trainer (Trainer): 
                The trainer used for training.

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
            >>> nnUNet.configure_environment()
            >>> nnUNet.process_dataset()
            >>> nnUNet.train_nnUNet()
        """

        # Define the command to train the nnUNet model
        self.logger.info(f"Training nnUNet model for fold{self.fold.value}...")
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
            self.nnUNet_preprocessed (str): 
                The path to the preprocessed data directory.

            self.dataset_id (str): 
                The ID of the dataset.

            self.dataset_name (str): 
                The name of the dataset.

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
            >>> nnUNet.configure_environment()
            >>> nnUNet.process_dataset()
        """

        self.logger.info("Processing dataset...")
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
            self.nnUNet_raw (str): 
                The path to the raw data directory.

            self.nnUNet_preprocessed (str): 
                The path to the preprocessed data directory.

            self.nnUNet_results (str): 
                The path to the results directory.

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
            >>> nnUNet.configure_environment()
        """

        # Create the necessary directories for nnUNet
        self.logger.info("Configuring nnUNet environment...")
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
        shutil.copy(CONFIG_NNUNET, f"{self.nnUNet_datapath}/dataset.json")
    
def _remove_files(folder_path: str):
    """Removes all files in the folder that start with 'BRATS_'."""
    for file in os.listdir(folder_path):
        if file.startswith("BRATS_"):
            os.remove(f"{folder_path}/{file}")

def _remove_folder(folder_path):
    """Deletes the specified folder and all its contents if it exists."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
