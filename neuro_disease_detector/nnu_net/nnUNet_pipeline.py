import subprocess, shutil, os
from neuro_disease_detector.nnu_net.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.utils.utils_dataset import split_assign
from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

def nnUNet_init(dataset_id: str, configuration: Configuration, fold: Fold, trainer: Trainer):
    """
    Initialize the nnUNet model using the specified dataset and configuration.

    Args:
        dataset_id (str): The ID of the dataset.
        configuration (Configuration): The configuration to use for training.
        fold (Fold): The fold to use for training.
        trainer (Trainer): The trainer to use for training.

    Returns:
        None

    Example:
        >>> from neuro_disease_detector.nnu_net.nnUNet_pipeline import nnUNet_init
        >>> from neuro_disease_detector.nnu_net.__init__ import Configuration, Fold, Trainer
        >>>
        >>> # Dataset ID
        >>> dataset_id = "024"
        >>>
        >>> # Configuration to use for training
        >>> configuration = Configuration.FULL_3D
        >>> 
        >>> # Fold to use for training
        >>> fold = Fold.FOLD_1
        >>> 
        >>> # Trainer to use for training
        >>> trainer = Trainer.EPOCHS_20
        >>> 
        >>> # Initialize and train the nnUNet model
        >>> nnUNet_init(dataset_id, configuration, fold, trainer)
    """
    
    nnunet_cwd = f"{cwd}/nnUNet"
    dataset_dir  = f"{nnunet_cwd}/MSLesSeg-Dataset"
    dataset_name = f"Dataset{dataset_id}_MSLesSeg"
    nnUNet_datapath = f"{nnunet_cwd}/nnUNet_raw/{dataset_name}"
    raw_data = f"{nnunet_cwd}/nnUNet_raw"
    process_data = f"{nnunet_cwd}/nnUNet_preprocessed"
    train_results = f"{nnunet_cwd}/nnUNet_results"
    test_results = f"{nnUNet_datapath}/nnUNet_tests_{fold.value}"

    logger.info(f"Downloading MSLesSeg-Dataset for nnUNet pipeline {dataset_id}...")
    url = "https://drive.google.com/uc?export=download&id=1A3ZpXHe-bLpaAI7BjPTSkZHyQwEP3pIi"
    download_dataset_from_cloud(dataset_dir, url)

    logger.info("Creating nnUNet dataset...")
    create_nnu_dataset(dataset_dir, nnUNet_datapath)

    logger.info("Configuring nnUNet environment...")
    configure_environment(raw_data, process_data, train_results)  

    logger.info("Processing dataset...")
    process_dataset(process_data, dataset_name, dataset_id)

    logger.info(f"Training nnUNet model for fold{fold.value}...")
    train_nnUNet(dataset_id, configuration, fold, trainer)

    logger.info("Performing inference on test data...")
    inference_test(nnUNet_datapath, test_results, dataset_id, configuration, trainer, fold)

    logger.info("Evaluating test results...")
    evaluate_test_results(nnUNet_datapath, test_results)
    logger.info("nnUNet pipeline completed.")

def evaluate_test_results(nnUNet_datapath: str, test_results: str):
    """
    Evaluate the test results using the trained nnUNet model.

    Args:
        nnUNet_datapath (str): The path to the nnUNet dataset.
        test_results (str): The path to the test results.

    Returns:
        None
    """

    test_path = f"{nnUNet_datapath}/labelsTs"

    # Define the command to evaluate the test results
    command = ["nnUNetv2_evaluate_folder", 
               test_path,
               test_results, 
               "-djfile", f"{test_results}/dataset.json", 
               "-pfile", f"{test_results}/plans.json"]
    
    subprocess.run(command, env=os.environ, check=True)

def inference_test(nnUNet_datapath: str, test_results: str, dataset_id: str, 
                   configuration: Configuration = Configuration.FULL_3D, 
                   trainer: Trainer = Trainer.EPOCHS_20,
                   fold: Fold = Fold.FOLD_1):
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
    os.makedirs(test_results, exist_ok=True)
    test_path = f"{nnUNet_datapath}/imagesTs"

    # Define the command to perform inference on the test data
    command = ["nnUNetv2_predict", 
               "-i", test_path, 
               "-o", test_results, 
               "-d", dataset_id,
               "-c", configuration.value, 
               "-tr", trainer.value,
               "-f", fold.value]
    
    subprocess.run(command, env=os.environ, check=True)

def train_nnUNet(dataset_id: str, 
                 configuration: Configuration = Configuration.FULL_3D, 
                 fold: Fold = Fold.FOLD_1, 
                 trainer: Trainer = Trainer.EPOCHS_20):
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
    command = ["nnUNetv2_train", 
               dataset_id, 
               configuration.value, 
               fold.value, 
               "-tr", trainer.value]
    
    subprocess.run(command, env=os.environ, check=True)
    
def process_dataset(process_data: str, dataset_name: str, dataset_id: str):
    """
    Process the dataset using nnUNet.

    Args:
        process_data (str): The path to the preprocessed data directory.
        dataset_name (str): The name of the dataset.
        dataset_id (str): The ID of the dataset.

    Returns:
        None
    """

    if os.path.exists(f"{process_data}/{dataset_name}"):
        return
    
    # Define and run the command to preprocess the dataset
    command = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity", "-np", "1"]
    subprocess.run(command, env=os.environ, check=True)

    # Copy the splits_final.json file to the appropriate directory
    shutil.copy(f"{cwd}/config/splits_final.json", f"{process_data}/{dataset_name}/splits_final.json")

def configure_environment(raw_data: str, process_data: str, results: str):
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
    os.makedirs(raw_data, exist_ok=True)
    os.makedirs(process_data, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    # Set the environment variables for nnUNet
    os.environ["nnUNet_raw"] = raw_data
    os.environ["nnUNet_preprocessed"] = process_data
    os.environ["nnUNet_results"] = results

def create_nnu_dataset(dataset_dir: str, nnUNet_datapath: str):
    """
    Create the nnUNet dataset from the MSLesSeg-Dataset.

    Args:
        dataset_dir (str): The path to the MSLesSeg-Dataset.
        nnUNet_datapath (str): The path to the nnUNet dataset.

    Returns:
        None
    """

    if os.path.exists(nnUNet_datapath):
        return
    
    # Define the path where the nnUNet dataset will be stored
    dataset_path = f"{dataset_dir}/train"
    
    # Create necessary directories for the nnUNet dataset
    os.makedirs(nnUNet_datapath)
    os.makedirs(f"{nnUNet_datapath}/imagesTr")  
    os.makedirs(f"{nnUNet_datapath}/imagesTs")  
    os.makedirs(f"{nnUNet_datapath}/labelsTr")  
    os.makedirs(f"{nnUNet_datapath}/labelsTs")  
    
    # Initialize a unique id counter for each subject
    id = 0

    # Iterate over the subjects in the dataset
    for pd in range(1, 54):
        # Skip the subject with id 30
        if pd == 30:
            continue

        # Define the path for a specific subject's folder
        pd_path = f"{dataset_path}/P{pd}"

        # Iterate over the 4 timepoints for each subject
        for td in range(1, 5):
            # Define the path for the timepoint folder
            td_path = f"{pd_path}/T{td}"

            # Break the loop if the timepoint folder doesn't exist
            if not os.path.exists(td_path):
                break

            # Increment the dataset ID for each subject/timepoint pair
            id += 1

            # Define the paths to the image and mask files
            flair_path = f"{td_path}/P{pd}_T{td}_FLAIR.nii"
            t1_path = f"{td_path}/P{pd}_T{td}_T1.nii"
            t2_path = f"{td_path}/P{pd}_T{td}_T2.nii"
            mask_path = f"{td_path}/P{pd}_T{td}_MASK.nii"

            # Assign the patient to a fold based on their ID
            train_test = "Ts" if split_assign(pd) == "Test" else "Tr"  

            # Copy the images and mask to the appropriate directories
            shutil.copy(flair_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0000.nii.gz")
            shutil.copy(t1_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0001.nii.gz")
            shutil.copy(t2_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0002.nii.gz")
            shutil.copy(mask_path, f"{nnUNet_datapath}/labels{train_test}/BRATS_{id}.nii.gz")

    # Copy the dataset.json file to the nnUNet dataset directory
    shutil.copy(f"{cwd}/config/dataset.json", f"{nnUNet_datapath}/dataset.json")

if __name__ == "__main__":
    dataset_id = "024"
    configuration = Configuration.FULL_3D
    fold = Fold.FOLD_1
    trainer = Trainer.EPOCHS_100
    nnUNet_init(dataset_id, configuration, fold, trainer)
