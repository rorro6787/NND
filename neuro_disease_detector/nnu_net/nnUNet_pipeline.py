import subprocess
import shutil
import os

from neuro_disease_detector.nnu_net.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.nnu_net.utils import get_patient_by_test_id
cwd = os.getcwd()

def nnUNet_init(dataset_dir: str, dataset_id, configuration: Configuration, fold: Fold, trainer: Trainer):
    """
    Initialize the nnUNet model using the specified dataset and configuration.

    Args:
        dataset_dir (str): The path to the dataset directory.
        dataset_id (str): The ID of the dataset.
        configuration (Configuration): The configuration to use for training.
        fold (Fold): The fold to use for training.
        trainer (Trainer): The trainer to use for training.

    Returns:
        None

    Example:
        >>> from neuro_disease_detector.nnu_net.nnUNet_pipeline import nnUNet_init
        >>> 
        >>> # Data source
        >>> dataset_path = f"{cwd}"
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
        >>> nnUNet_init(dataset_path, dataset_id, configuration, fold, trainer)
    """
    
    dataset_name = f"Dataset{dataset_id}_MSLesSeg"
    nnUNet_datapath = f"{cwd}/nnUNet_raw/{dataset_name}"
    
    raw_data = f"{cwd}/nnUNet_raw"
    process_data = f"{cwd}/nnUNet_preprocessed"
    results = f"{cwd}/nnUNet_results"

    create_nnu_dataset(dataset_dir, nnUNet_datapath)
    configure_environment(raw_data, process_data, results)  
    process_dataset(process_data, dataset_name, dataset_id)
    train_nnUNet(dataset_id, configuration, fold, trainer)

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
    command = ["nnUNetv2_train", dataset_id, configuration.value, fold.value, "-tr", trainer.value]
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

    if not os.path.exists(f"{process_data}/{dataset_name}"):
        # Define and run the command to preprocess the dataset
        command = ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity", "-np", "1"]
        subprocess.run(command, env=os.environ, check=True)

        # Copy the splits_final.json file to the appropriate directory
        shutil.copy(f"{cwd}/splits_final.json", f"{process_data}/{dataset_name}/splits_final.json")

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

    # Define the path where the nnUNet dataset will be stored
    dataset_path = f"{dataset_dir}/MSLesSeg-Dataset/train"

    if os.path.exists(nnUNet_datapath):
        return
    
    # Create necessary directories for the nnUNet dataset
    os.makedirs(nnUNet_datapath)
    os.makedirs(f"{nnUNet_datapath}/imagesTr")  # Training images folder
    os.makedirs(f"{nnUNet_datapath}/imagesTs")  # Testing images folder
    os.makedirs(f"{nnUNet_datapath}/labelsTr")  # Training labels folder
    os.makedirs(f"{nnUNet_datapath}/labelsTs")  # Testing labels folder
    
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

            # Assign the subject to either the 'train' or 'test' fold
            fold = _split_assign(pd)  # Function to assign fold (train/test)
            train_test = "Ts" if fold == "test" else "Tr"  # Test or Train based on fold

            # Copy the images and mask to the appropriate directories
            shutil.copy(flair_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0000.nii.gz")
            shutil.copy(t1_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0001.nii.gz")
            shutil.copy(t2_path, f"{nnUNet_datapath}/images{train_test}/BRATS_{id}_0002.nii.gz")
            shutil.copy(mask_path, f"{nnUNet_datapath}/labels{train_test}/BRATS_{id}.nii.gz")

    shutil.copy(f"{cwd}/dataset.json", f"{nnUNet_datapath}/dataset.json")

def _split_assign(pd: int):
    """
    Assign a patient to a fold based on the patient ID.

    Args:
        pd (int): The patient ID.

    Returns:
        str: The fold to which the patient belongs.
    """

    # Define the boundaries for each fold
    folds = [1, 6, 12, 19, 28, 41]

    # Assign the patient to a fold based on their ID
    for i, start in enumerate(folds[:-1]):
        # If the patient ID is within the range of the current fold, return the fold
        if pd >= start and pd < folds[i + 1]:
            return f"fold{i + 1}"
    # If the patient ID is not within the range of any fold, return "test"
    return "test"

if __name__ == "__main__":
    dataset_path = f"{cwd}"
    dataset_id = "024"
    configuration = Configuration.FULL_3D
    fold = Fold.FOLD_1
    trainer = Trainer.EPOCHS_20
    
    nnUNet_init(dataset_path, dataset_id, configuration, fold, trainer)
    # print(get_patient_by_test_id("30"))
