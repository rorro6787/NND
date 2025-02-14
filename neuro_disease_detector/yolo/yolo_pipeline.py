import os 

from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud
from neuro_disease_detector.yolo.validation_consensus import YoloFoldValidator
from neuro_disease_detector.yolo.process_dataset import process_dataset
from neuro_disease_detector.yolo.train_augm import train_yolo_folds
from neuro_disease_detector.yolo.__init__ import YoloModel
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

def yolo_init(yolo_model: YoloModel, id: str, consensus_threshold: int=2) -> None:
    """
    Initialize the YOLO dataset processing pipeline.

    Args:
        yolo_model (YoloModel): The YOLO model to be used for the detection task.
        id (str): ID for the training fold
        consensus_threshold (int): Consensus value for prediction voting. Goes from 1 to 3

    Returns:
        None

    Example:
        >>> from neuro_disease_detector.yolo.process_dataset import yolo_init
        >>>
        >>> # Initialize the YOLO dataset processing pipeline
        >>> nnUNet_init(dataset_id, configuration, fold, trainer)
    """

    dataset_dir = f"{cwd}/MSLesSeg-Dataset"
    yolo_dataset = f"{cwd}/MSLesSeg-Dataset-a"

    logger.info(f"Downloading MSLesSeg-Dataset for yolo pipeline...")
    url = "https://drive.google.com/uc?export=download&id=1A3ZpXHe-bLpaAI7BjPTSkZHyQwEP3pIi"
    download_dataset_from_cloud(dataset_dir, url)

    logger.info("Creating and processing YOLO dataset...")
    # process_dataset(dataset_dir, yolo_dataset)
    url_yolo = "https://drive.google.com/uc?export=download&id=1g6g2Oe2kPYgt7pn-KSgv170aN4T8ZYbi"
    download_dataset_from_cloud(yolo_dataset, url_yolo, extract_folder=False)
    
    logger.info(f"Training yolo model for...")
    train_path = train_yolo_folds(id, yolo_model, cwd)

    """
    logger.info("Evaluating test results...")
    yolo_fold_validator = YoloFoldValidator(train_path, cwd, consensus_threshold=consensus_threshold)
    yolo_fold_validator.validate_all_folds()
    
    cm_fold_epoch = yolo_fold_validator.cm_fold_epoch
    metrics_fold_epoch = yolo_fold_validator.metrics_fold_epoch
    print(cm_fold_epoch, metrics_fold_epoch)
    """
    
    logger.info("yolo pipeline completed.")
    

if __name__ == "__main__":
    yolo_model = YoloModel.V11M_SEG
    consensus_threshold = 2
    id = "024"
    yolo_init(yolo_model, id, consensus_threshold=2)
