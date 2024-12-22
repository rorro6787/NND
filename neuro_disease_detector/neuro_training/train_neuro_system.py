from ultralytics import YOLO
from pathlib import Path
from neuro_disease_detector.utils.utils_training import generate_yaml_files
import yaml
import os

from neuro_disease_detector.logger import get_logger
logger = get_logger(__name__)

def train_neuro_system(yolo_model: str, model_name: str, yaml_file_path: str) -> None:
    """
    Trains a YOLO model for image segmentation using a specified dataset.

    Parameters:
        model_name (str): The name of the experiment or model.
        yaml_file_path (str): Path to the YAML file containing data configuration for training.

    Returns:
        None
    """

    # Load the pre-trained YOLO model for segmentation
    model = YOLO(yolo_model, task="segmentation")

    # Define the directory to save training results
    save_directory = Path("runs")

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Train the model with the specified dataset and parameters
    model.train(
        data=yaml_file_path,       # Path to the YAML file with data configuration
        epochs=256,                # Number of training epochs
        imgsz=320,                 # Image size (width and height)
        batch=-1,                  # Batch size, -1 for default
        name=model_name,           # Experiment/model name
        device=0,                  # Device ID for training (0 for first GPU)
        project=save_directory,    # Project directory for results
        save_dir=save_directory,   # Directory to save the trained model
        fraction=1,                # Fraction of dataset for training
        plots=True,                # Generate training plots
        save_period=3,             # Save model every 'n' epochs
        dropout=0.25,              # Dropout rate for regularization
        patience=20,               # Early stopping patience (number of epochs)
        resume=False,              # Whether to resume training from the last checkpoint
        pretrained=True,           # Use pretrained weights
        # freeze=[0],              # Freeze specific layers (list of layer indices)
        # hyp=None,                # Hyperparameter file path or None for defaults
        # local_rank=-1,           # Local GPU rank for distributed training
        # sync_bn=False,           # Use synchronized batch norm
        # workers=8,               # Number of data loading workers
        # lr0=0.01,                # Initial learning rate
        # lrf=0.001,               # Final learning rate
        # weight_decay=0.0005,     # Weight decay for regularization
        # momentum=0.937,          # Momentum for SGD optimizer
        # dampening=0.5,           # Momentum damping
        # nesterov=True,           # Use Nesterov momentum
        # accumulative=2,          # Gradient accumulation steps
        # optimizer="AdamW",
        # amp=False,
        # val=True,                # Validate model after each epoch
        # image_weights=False,     # Weight images in loss
        # hyp_path=None,           # Path to hyperparameter file
        # save_json=True,          # Save results as JSON
        # lr_schedule=True,        # Use learning rate scheduling
        # rect=False,              # Use rectangular image resizing
        # single_cls=False,        # Train on single class only
        # compute_map=False,       # Calculate mAP during validation
        # iou = 0.9,               # IoU threshold for mAP calculation
        # conf = 0.2,              # Confidence threshold for mAP calculation
    )

def train_neuro_system_k_folds(dataset_path: str) -> None:
    """
    Trains the YOLO segmentation model using 5-fold cross-validation.

    This function iterates through 5 folds, generating a model name and corresponding
    YAML configuration file for each fold, then calls `train_neuro_system` to train the model.

    Parameters:
        None

    Returns:
        None
    """

    # Select the disere YOLO model for K-fold training process
    yolo_model = "yolo11n-seg.pt"

    yaml_files = generate_yaml_files(dataset_path)
    os.makedirs(Path("k_fold_configs"), exist_ok=True)

    for index, yaml_data in enumerate(yaml_files):
        # Save the YAML configuration for this fold
        yaml_file_path = os.path.join("k_fold_configs", f"MSLesSeg_Dataset-{index+1}.yaml")
        
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        name_model = f"yolov11n-seg-me-kfold-{index+1}"
        logger.info(f"Training model {index+1}")
        train_neuro_system(yolo_model, name_model, yaml_file_path)

if __name__ == "__main__":
    # dataset_path = process_dataset()
    dataset_path = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/data_processing"
    train_neuro_system_k_folds(dataset_path)
