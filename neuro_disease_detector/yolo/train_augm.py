from ultralytics import YOLO
import yaml
import os

from neuro_disease_detector.yolo.__init__ import YoloModel
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

def _train_parameters(**train_params) -> dict:
    """
    Define the training parameters for the YOLO model.

    Args:
        **train_params: Custom training parameters to override the defaults.

    Returns:
        dict: A dictionary containing the training parameters.

    Example:
        >>> from neuro_disease_detector.yolo.neuro_training.train_neuro_system import _train_parameters
        >>>
        >>> # Get the training parameters for the YOLO model
        >>> train_params = _train_parameters(epochs=256, imgsz=320, batch=-1, device=0, fraction=1, plots=True, save_period=3, dropout=0.25, patience=20, resume=False, pretrained=True)
        >>> print(train_params)
        {'epochs': 256, 'imgsz': 320, 'batch': -1, 'device': 0, 'fraction': 1, 'plots': True, 'save_period': 3, 'dropout': 0.25, 'patience': 20, 'resume': False, 'pretrained': True}
    """

    # Define the training parameters for the YOLO model
    train_parameters = {
        "epochs" : 100,                # Number of training epochs
        "imgsz" : 320,                 # Image size (width and height)
        "batch" : -1,                  # Batch size, -1 for default
        "device" : 0,                  # Device ID for training (0 for first GPU)
        "fraction" : 1,                # Fraction of dataset for training
        "plots" : True,                # Generate training plots
        "save_period" : 100,           # Save model every 'n' epochs
        "dropout" : 0.25,              # Dropout rate for regularization
        "patience" : 20,               # Early stopping patience (number of epochs)
        "resume" : False,              # Whether to resume training from the last checkpoint
        "pretrained" : True,           # Use pretrained weights
        # "freeze" : [0],                # Freeze specific layers (list of layer indices)
        # "hyp" : None,                  # Hyperparameter file path or None for defaults
        # "local_rank" : -1,             # Local GPU rank for distributed training
        # "sync_bn" : False,             # Use synchronized batch norm
        # "workers" : 8,                 # Number of data loading workers
        # "lr0" : 0.01,                  # Initial learning rate
        # "lrf" : 0.001,                 # Final learning rate
        # "weight_decay" : 0.0005,       # Weight decay for regularization
        # "momentum" : 0.937,            # Momentum for SGD optimizer
        # "dampening" : 0.5,             # Momentum damping
        # "nesterov" : True,             # Use Nesterov momentum
        # "accumulative" : 2,            # Gradient accumulation steps
        # "optimizer" : "AdamW",         # Optimizer for training
        # "amp" : False,                 # Use automatic mixed precision
        # "val" : True,                  # Validate model after each epoch
        # "image_weights" : False,       # Weight images in loss
        # "hyp_path" : None,             # Path to hyperparameter file
        # "save_json" : True,            # Save results as JSON
        # "lr_schedule" : True,          # Use learning rate scheduling
        # "rect" : False,                # Use rectangular image resizing
        # "single_cls" : False,          # Train on single class only
        # "compute_map" : False,         # Calculate mAP during validation
        # "iou" : 0.9,                   # IoU threshold for mAP calculation
        # "conf" : 0.2,                  # Confidence threshold for mAP calculation
    }

    # Update default parameters with the custom ones passed as keyword arguments
    train_parameters.update(**train_params)
    return train_parameters

def _augmentation_parameters(**augmentation_params) -> dict:
    """
    Define the augmentation parameters for the YOLO model.

    Args:
        **augmentation_params: Custom augmentation parameters to override the defaults.

    Returns:
        dict: A dictionary containing the augmentation parameters.

    Example:
        >>> from neuro_disease_detector.yolo.neuro_training.train_neuro_system import _augmentation_parameters
        >>>
        >>> # Get the augmentation parameters for the YOLO model
        >>> augmentation_params = _augmentation_parameters(augment=True, mosaic=True, mixup=True, copy_paste=True, cutmix=True, random_shapes=True, autoanchor=True, sync_bn=True)
        >>> print(augmentation_params)
        {'augment': True, 'mosaic': True, 'mixup': True, 'copy_paste': True, 'cutmix': True, 'random_shapes': True, 'autoanchor': True, 'sync_bn': True}
    """

    # Define the augmentation parameters for the YOLO model
    augmentation_parameters = {
        "hsv_h" : 0.015,                         # Range: (0.0 - 1.0)         | Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        "hsv_s" : 0.7,                           # Range: (0.0 - 1.0)         | Alters the saturation of the image by a fraction, affecting the intensity of colors. Useful for simulating different environmental conditions.
        "hsv_v" : 0.4,                           # Range: (0.0 - 1.0)         | Modifies the value (brightness) of the image by a fraction, helping the model to perform well under various lighting conditions.
        "degrees" : 0.0,                         # Range: (-180 - +180)       | Rotates the image randomly within the specified degree range, improving the model's ability to recognize objects at various orientations.
        "translate" : 0.1,                       # Range: (0.0 - 1.0)         | Translates the image horizontally and vertically by a fraction of the image size, aiding in learning to detect partially visible objects.
        "scale" : 0.5,                           # Range: (>=0.0)             | Scales the image by a gain factor, simulating objects at different distances from the camera.
        "shear" : 0.0,                           # Range: (-180 - +180)       | Shears the image by a specified degree, mimicking the effect of objects being viewed from different angles.
        "perspective" : 0.0,                     # Range: (0.0 - 0.001)       | Applies a random perspective transformation to the image, enhancing the model's ability to understand objects in 3D space.
        "flipud" : 0.0,                          # Range: (0.0 - 1.0)         | Flips the image upside down with the specified probability, increasing the data variability without affecting the object's characteristics.
        "fliplr" : 0.5,                          # Range: (0.0 - 1.0)         | Flips the image left to right with the specified probability, useful for learning symmetrical objects and increasing dataset diversity.
        "bgr" : 0.0,                             # Range: (0.0 - 1.0)         | Flips the image channels from RGB to BGR with the specified probability, useful for increasing robustness to incorrect channel ordering.
        "mosaic" : 1.0,                          # Range: (0.0 - 1.0)         | Combines four training images into one, simulating different scene compositions and object interactions. Highly effective for complex scene understanding.
        "mixup" : 0.0,                           # Range: (0.0 - 1.0)         | Blends two images and their labels, creating a composite image. Enhances the model's ability to generalize by introducing label noise and visual variability.
        "copy_paste" : 0.0,                      # Range: (0.0 - 1.0)         | Copies and pastes objects across images, useful for increasing object instances and learning object occlusion. Requires segmentation labels.
        "copy_paste_mode" : "flip",              # Range: ("flip", "mixup")   | Copy-Paste augmentation method selection among the options of ("flip", "mixup").
        "auto_augment" : "randaugment",          # Range: ("randaugment",     | Automatically applies a predefined augmentation policy (randaugment, autoaugment, augmix), optimizing for classification tasks by diversifying the visual features.
                                                 #         "autoaugment", 
                                                 #         "augmix")          
        "erasing" : 0.4,                         # Range: (0.0 - 0.9)         | Randomly erases a portion of the image during classification training, encouraging the model to focus on less obvious features for recognition.
        "crop_fraction" : 1.0,                   # Range: (0.1 - 1.0)         | Crops the classification image to a fraction of its size to emphasize central features and adapt to object scales, reducing background distractions.
    }
    
    # Update default parameters with the custom ones passed as keyword arguments
    augmentation_parameters.update(**augmentation_params)
    return augmentation_parameters

def train_yolo(yolo_model: YoloModel, yaml_file_path: str, train_path: str, fold: str) -> None:
    """
    Trains a YOLO model for image segmentation using a specified dataset.

    Args:
        model_name (YoloModel): The name of the experiment or model.
        yaml_file_path (str): Path to the YAML file containing data configuration for training.
        train_path (str): The path to the directory where the training results will be saved.
        fold (str): The fold name for the current training iteration.

    Returns:
        None
    """

    # Define the training and augmentation parameters for the YOLO model
    train_params = _train_parameters()
    augmentation_params = _augmentation_parameters()
    params = {**train_params, **augmentation_params}

    # Train the model with the specified dataset and parameters
    model = YOLO(yolo_model.value, task="segmentation")
    model.train(
        data=yaml_file_path,        # Path to the YAML file with data configuration
        project=train_path,         # Project directory for results
        save_dir=train_path,        # Directory to save the trained model
        name=fold,                  # Experiment/model name
        **params                    # Training parameters
    )

def train_yolo_folds(id: str, yolo_model: YoloModel, dataset_path: str) -> str:
    """
    Trains the YOLO segmentation model using 5-fold cross-validation. This function iterates through 5 folds, generating a model name and corresponding YAML configuration file for each fold, then calls `train_neuro_system` to train the model.

    Args:
        id (str): ID for the training fold
        yolo_model (YoloModel): The YOLO model to train.
        dataset_path (str): The path to the dataset directory.

    Returns:
        train_path (str): The path where the training results are stored.

    Example:
        >>> from neuro_disease_detector.yolo.train_augm import train_yolo_folds
        >>> from neuro_disease_detector.yolo.__init__ import YoloModel
        >>>
        >>> # Define the path to the dataset directory
        >>> dataset_path = "/path/to/dataset"
        >>>
        >>> # Train the YOLO model using 5-fold cross-validation
        >>> train_yolo_folds(YoloModel.V8N_SEG, dataset_path)
    """

    # Define the model name for the YOLO model and create the YAML configuration files
    yolo_model_suffix = yolo_model.value.removesuffix(".pt")
    train_path = f"{cwd}/yolo_trainings/{id}/{yolo_model_suffix}"
    yaml_files = _generate_yaml_files(dataset_path)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(f"{train_path}/config", exist_ok=True)
    
    for index, yaml_data in enumerate(yaml_files):
        # Save the YAML configuration for this fold
        yaml_file_path = f"{train_path}/config/fold{index+1}.yaml"
        
        # Write the YAML data to the file
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)

        # Define the model name for this fold and train the model
        fold = f"fold{index+1}"
        logger.info(fold)
        train_yolo(yolo_model, yaml_file_path, train_path, fold)

    return train_path

def _generate_yaml_files(dataset_path: str) -> list:
    """Generate YAML configuration files for each fold in a 5-fold cross-validation setup."""

    # Define the folds for the dataset
    fold_configs = []
    k = 5
    
    # Create a mapping for validation folds
    val_mapping = { 1:5, 2:4, 3:3, 4:2, 5:1 }
    
    # Iterate over each fold to create train, val splits
    for i in range(1, k + 1):
        # Initialize the data configuration for this fold
        data = {'train': [], 'val': '', 'nc': 1, 'names': ['multiple_esclerosis']}
        
        # Assign validation fold based on mapping
        val_fold = val_mapping[i]
        data['val'] = f"{dataset_path}/MSLesSeg-Dataset-a/fold{val_fold}/images"
        
        # Append the training data for all folds except the validation fold
        for j in range(1, k + 1):
            if j != val_fold:
                data['train'].append(f"{dataset_path}/MSLesSeg-Dataset-a/fold{j}/images")
        
        # Append this iteration's configuration to the list
        fold_configs.append(data)
    
    return fold_configs
    