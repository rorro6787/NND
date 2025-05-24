import os 

from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud
from neuro_disease_detector.models.yolo.validation_consensus import YoloFoldValidator
from neuro_disease_detector.models.yolo.process_dataset import process_dataset
from neuro_disease_detector.models.yolo.train_augm import YoloFoldTrainer
from neuro_disease_detector.models.yolo.__init__ import YoloModel, Trainer, Validator
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

def yolo_init(id: str, yolo_model: YoloModel, trainer: Trainer, validator: Validator, consensus_threshold: int = 2) -> None:
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
    
    download_dataset_from_cloud(dataset_dir, 
                                "https://drive.google.com/uc?export=download&id=1TM4ciSeiyl-ri4_Jn4-aMOTDSSSHM6XB", 
                                extract_folder=False
    )
    
    # process_dataset(dataset_dir, yolo_dataset)
    download_dataset_from_cloud(yolo_dataset, 
                                "https://drive.google.com/uc?export=download&id=1sFl9kNsN4jShUACiwvKzBP9T_Q-kXSc4", 
                                extract_folder=True
    )
    
    logger.info(f"Training yolo model for...")
    yolo_fold_trainer = YoloFoldTrainer(id, yolo_model, trainer, cwd)
    yolo_fold_trainer.train_k_fold()
    
    logger.info("Evaluating test results...")

    """
    yolo_fold_validator = YoloFoldValidator(train_path, cwd, consensus_threshold=consensus_threshold)
    yolo_fold_validator.validate_all_folds()
    
    print(yolo_fold_validator.cm_fold_epoch)
    print(yolo_fold_validator.metrics_fold_epoch)
    """
    
    logger.info("yolo pipeline completed.")
    
if __name__ == "__main__":
    yolo_model = YoloModel.V11X_SEG
    validator = Validator.A2D
    trainer = Trainer.FULL_3D
    for i in range(5):
        id = f"00{i}"
        yolo_init(id, yolo_model, trainer, validator, consensus_threshold=2)
