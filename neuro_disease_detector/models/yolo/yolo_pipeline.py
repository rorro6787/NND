import os
from neuro_disease_detector.utils.utils_dataset import download_dataset_from_cloud
from neuro_disease_detector.models.yolo.validation_consensus import YoloFoldValidator
from neuro_disease_detector.models.yolo.process_dataset import process_dataset
from neuro_disease_detector.models.yolo.train_augm import YoloFoldTrainer
from neuro_disease_detector.models.yolo.__init__ import YoloModel, Trainer, Validator
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

def yolo_init(
    fold_id: str,
    yolo_model: YoloModel,
    trainer: Trainer,
    validator: Validator,
    consensus_threshold: int = 2,
) -> None:
    """
    Initialize the YOLO dataset processing, training, and validation pipeline.

    Args:
        fold_id (str):   Identifier for this particular training run (e.g. "000", "001", ...).
        yolo_model (YoloModel):   Which YOLO backbone/configuration to use.
        trainer (Trainer):         Which training regime (e.g. FULL_3D, etc.).
        validator (Validator):     Which Validator enum to use (e.g. A2D, Cs3D, etc.).
        consensus_threshold (int): For multi‐plane consensus (only matters if validator is Cs2D or Cs3D).

    Structure of the pipeline:
      1. Download (raw) MSLesSeg dataset into `dataset_dir`.
      2. Download or generate YOLO‐formatted dataset into `yolo_dataset`.
      3. Train k‐fold YOLO models via YoloFoldTrainer.
      4. Validate the k‐fold weights via YoloFoldValidator and print out confusion matrices & metrics.
    """

    # 1) Define paths for raw and YOLO‐formatted datasets:
    dataset_dir = os.path.join(cwd, "MSLesSeg-Dataset")       # "raw" MSLesSeg data (NIfTI files, etc.)
    yolo_dataset = os.path.join(cwd, "MSLesSeg-Dataset-YOLO")  # YOLO‐formatted PNGs + labels

    # 2) Download the raw MSLesSeg dataset (if not already present):
    if not os.path.isdir(dataset_dir):
        logger.info(f"Downloading raw MSLesSeg dataset into {dataset_dir} …")
        download_dataset_from_cloud(
            dataset_dir,
            "https://drive.google.com/uc?export=download&id=1TM4ciSeiyl-ri4_Jn4-aMOTDSSSHM6XB",
            extract_folder=False,
        )
    else:
        logger.info(f"Raw dataset already exists at {dataset_dir}, skipping download.")

    # 3) Prepare (or download) the YOLO‐formatted dataset:
    #    You have two options:
    #      a) If you want to generate the YOLO‐formatted dataset yourself, uncomment process_dataset(...).
    #      b) Otherwise, you can download a preprocessed YOLO dataset from the cloud (as below).
    #
    # Option (a) – uncomment if you have the raw data and want to run the processing step:
    #
    # if not os.path.isdir(yolo_dataset):
    #     logger.info(f"Generating YOLO‐formatted dataset in {yolo_dataset} …")
    #     process_dataset(dataset_dir, yolo_dataset)
    # else:
    #     logger.info(f"YOLO dataset already exists at {yolo_dataset}, skipping processing.")
    #
    # Option (b) – download a preprocessed YOLO dataset (comment out if you ran process_dataset yourself):
    if not os.path.isdir(yolo_dataset):
        logger.info(f"Downloading preprocessed YOLO dataset into {yolo_dataset} …")
        download_dataset_from_cloud(
            yolo_dataset,
            "https://drive.google.com/uc?export=download&id=1sFl9kNsN4jShUACiwvKzBP9T_Q-kXSc4",
            extract_folder=True,
        )
    else:
        logger.info(f"YOLO dataset already exists at {yolo_dataset}, skipping download.")

    # 4) Train k‐fold YOLO models:
    logger.info(f"Starting k-fold YOLO training (fold_id={fold_id}) …")
    yolo_fold_trainer = YoloFoldTrainer(fold_id, yolo_model, trainer, cwd)
    yolo_fold_trainer.train_k_fold()
    logger.info("Finished training all k-folds.")

    # 5) Validate the k-fold weights:
    #
    #    YoloFoldTrainer will have created a directory under `cwd` named "yolo_trainings/<fold_id>/…",
    #    where each subfolder "fold1", "fold2", … "foldk" contains its own `weights/best.pt`.
    #
    #    We assume here that the folds directory is:
    folds_directory = os.path.join(cwd, "yolo_trainings", fold_id)
    if not os.path.isdir(folds_directory):
        raise FileNotFoundError(
            f"Expected to find trained‐folds directory at {folds_directory} but it does not exist."
        )

    logger.info("Evaluating k-fold trained models on the raw dataset …")
    yolo_fold_validator = YoloFoldValidator(
        folds_directory=folds_directory,
        data_folder=dataset_dir,
        validator=validator,
        consensus_threshold=consensus_threshold,
        k_folds=5,
    )
    yolo_fold_validator.validate_all_folds()

    # 6) Print out confusion matrices and metrics per fold:
    print("\n=== Confusion matrices (per fold) ===")
    for fold_name, cm_dict in yolo_fold_validator.cm_fold_epoch.items():
        print(f"{fold_name}: {cm_dict}")
    print("\n=== Metrics (per fold) ===")
    for fold_name, metrics_dict in yolo_fold_validator.metrics_fold_epoch.items():
        print(f"{fold_name}:")
        for metric_name, value in metrics_dict.items():
            print(f"  {metric_name.name}: {value:.4f}")
    logger.info("Validation complete. YOLO pipeline done.")


if __name__ == "__main__":
    # Example: Train and validate 5 separate runs, each with its own fold_id = "000", "001", …:
    yolo_model = YoloModel.V11X_SEG   # e.g. the YOLOv11x segmentation model
    validator = Validator.A2D         # e.g. single‐plane axial 2D validator
    trainer = Trainer.FULL_3D         # e.g. full 3D training regime

    # Run five independent trains/validations (IDs "000" … "004"):
    for i in range(5):
        fold_id = f"{i:03d}"  # "000", "001", "002", "003", "004"
        yolo_init(
            fold_id=fold_id,
            yolo_model=yolo_model,
            trainer=trainer,
            validator=validator,
            consensus_threshold=2,
        )
