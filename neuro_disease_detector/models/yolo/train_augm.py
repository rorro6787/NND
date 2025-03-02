from ultralytics import YOLO
import yaml
import os

from neuro_disease_detector.models.yolo.__init__ import YoloModel, Trainer, Validator
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)
cwd = os.getcwd()

class YoloFoldTrainer:
    """
    Class for training a YOLO model for image segmentation using a specified dataset with a 5-fold cross-validation setup.

    Attributes:
        id (str): 
            ID for the training fold.

        yolo_model (YoloModel): 
            The YOLO model to train.

        trainer (Trainer): 
            The training strategy to use.

        train_path (str):  
            The path to the directory where the training results will be saved.

        dataset_path (str): 
            The path to the dataset directory
        
        k (int):
            The number of folds for cross-validation.

        logger (Logger):
            The logger instance for the YOLO fold trainer.

    Methods:
        __init__(self, id: str, yolo_model: YoloModel, trainer: Trainer, dataset_path: str):
            Initializes the YOLO fold trainer with the specified ID, YOLO model, trainer, and dataset path.
        
        train_k_fold(self):
            Trains the YOLO model for image segmentation using a specified dataset with a 5-fold cross-validation setup.

        _train_yolo(self, yaml_file_path: str, fold: str):
            Trains a YOLO model for image segmentation using a specified dataset.

        _generate_yaml_files(self) -> list:
            Generate YAML configuration files for each fold in a 5-fold cross-validation setup.
    """
    
    def __init__(self, id: str, yolo_model: YoloModel, trainer: Trainer, dataset_path: str) -> None:
        """
        Initialize the YOLO fold trainer.

        Args:
            id (str): 
                ID for the training fold

            yolo_model (YoloModel): 
                The YOLO model to train.

            trainer (Trainer): 
                The training strategy to use.

            dataset_path (str): 
                The path to the dataset directory.

        Returns:
            None

        Example:
            >>> from neuro_disease_detector.models.yolo.__init__ import YoloModel, Trainer
            >>> from neuro_disease_detector.models.yolo.train_augm import YoloFoldTrainer
            >>> import os
            >>> 
            >>> dataset_id = "024"
            >>> trainer = Trainer.FULL_3D
            >>> dataset_path = os.getcwd()
            >>> yolo_fold_trainer = YoloFoldTrainer(dataset_id, YoloModel.V11M_SEG, trainer, dataset_path)
        """

        # Define the model name for the YOLO model and create the YAML configuration files
        self.id = id
        self.yolo_model = yolo_model.value.removesuffix(".pt")
        self.trainer = trainer
        self.train_path = f"{cwd}/yolo_trainings/{id}/{self.yolo_model}/{trainer.value}"
        self.dataset_path = dataset_path
        self.k = 5

        self.logger = get_logger(__name__)
        
    def train_k_fold(self) -> None:
        """
        Trains the YOLO model for image segmentation using a specified dataset with a 5-fold cross-validation setup.

        Args:
            self.train_path (str): 
                The path to the directory where the training results will be saved.

        Returns:
            None
        
        Example:
            >>> from neuro_disease_detector.models.yolo.__init__ import YoloModel, Trainer
            >>> from neuro_disease_detector.models.yolo.train_augm import YoloFoldTrainer
            >>> import os
            >>> 
            >>> dataset_id = "024"
            >>> trainer = Trainer.FULL_3D
            >>> dataset_path = os.getcwd()
            >>> yolo_fold_trainer = YoloFoldTrainer(dataset_id, YoloModel.V11M_SEG, trainer, dataset_path)
            >>> yolo_fold_trainer.train_k_fold()
        """

        # Generate the YAML configuration files for each fold
        yaml_files = self._generate_yaml_files()
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(f"{self.train_path}/config", exist_ok=True)
        
        for index, yaml_data in enumerate(yaml_files):
            # Save the YAML configuration for this fold
            yaml_file_path = f"{self.train_path}/config/fold{index+1}.yaml"
            
            # Write the YAML data to the file
            with open(yaml_file_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)

            # Define the model name for this fold and train the model
            fold = f"fold{index + 1}"
            self.logger.info(f"Training YOLO model for fold {fold}...")
            self._train_yolo(yaml_file_path, fold)

    def _train_yolo(self, yaml_file_path: str, fold: str) -> None:
        """Trains a YOLO model for image segmentation using a specified dataset."""

        # Define the training and augmentation parameters for the YOLO model
        train_params = _train_parameters()
        augmentation_params = _augmentation_parameters()
        params = {**train_params, **augmentation_params}

        if self.trainer == Trainer.FULL_3D or self.trainer == Trainer.SIMPLE_CORONAL:
            params["imgsz"] = 256
        
        """
        if fold == "fold1" and self.id == "001" and self.trainer == Trainer.SIMPLE_AXIAL:
            params["resume"] = True
            model = YOLO("/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/models/yolo/yolo_trainings/001/yolo11x-seg/axial/fold1/weights/last.pt", task="segmentation")
        else:
        """
        
        # Train the model with the specified dataset and parameters
        model = YOLO(self.yolo_model, task="segmentation")
        model.train(
            data=yaml_file_path,           # Path to the YAML file with data configuration
            project=self.train_path,       # Project directory for results
            save_dir=self.train_path,      # Directory to save the trained model
            name=fold,                     # Experiment/model name
            **params                       # Training parameters
        )

    def _generate_yaml_files(self) -> list:
        """Generate YAML configuration files for each fold in a 5-fold cross-validation setup."""

        # Define the folds for the dataset
        fold_configs = []
        
        # Create a mapping for validation folds
        val_mapping = { 1:5, 2:4, 3:3, 4:2, 5:1 }
        
        # Iterate over each fold to create train, val splits
        for i in range(1, self.k + 1):
            # Initialize the data configuration for this fold
            data = {'train': [], 'val': [], 'nc': 1, 'names': ['multiple_esclerosis']}
            
            # Assign validation fold based on mapping
            val_fold = val_mapping[i]
            if self.trainer == Trainer.FULL_3D:
                data['val'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{val_fold}/{Trainer.SIMPLE_AXIAL.value}/images")
                data['val'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{val_fold}/{Trainer.SIMPLE_CORONAL.value}/images")
                data['val'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{val_fold}/{Trainer.SIMPLE_SAGITTAL.value}/images")
            else:
                data['val'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{val_fold}/{self.trainer.value}/images")
            
            # Append the training data for all folds except the validation fold
            for j in range(1, self.k + 1):
                if j != val_fold:
                    if self.trainer == Trainer.FULL_3D:
                        data['train'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{j}/{Trainer.SIMPLE_AXIAL.value}/images")
                        data['train'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{j}/{Trainer.SIMPLE_CORONAL.value}/images")
                        data['train'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{j}/{Trainer.SIMPLE_SAGITTAL.value}/images")
                    else:
                        data['train'].append(f"{self.dataset_path}/MSLesSeg-Dataset-a/fold{j}/{self.trainer.value}/images")
            
            # Append this iteration's configuration to the list
            fold_configs.append(data)
        
        return fold_configs
            
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
        "epochs" : 100,                  # Number of training epochs
        "imgsz" : 192,                   # Image size (width and height)
        "batch" : -1,                    # Batch size, -1 for default
        "device" : 0,                    # Device ID for training (0 for first GPU)
        "fraction" : 1,                  # Fraction of dataset for training
        "plots" : True,                  # Generate training plots
        "save_period" : 75,              # Save model every 'n' epochs
        "dropout" : 0.2,                 # Dropout rate for regularization
        "patience" : 15,                 # Early stopping patience (number of epochs)
        "resume" : False,                # Whether to resume training from the last checkpoint
        "pretrained" : True,             # Use pretrained weights
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
        "erasing" : 0.0,                         # Range: (0.0 - 0.9)         | Randomly erases a portion of the image during classification training, encouraging the model to focus on less obvious features for recognition.
        "crop_fraction" : 1.0,                   # Range: (0.1 - 1.0)         | Crops the classification image to a fraction of its size to emphasize central features and adapt to object scales, reducing background distractions.
    }
    
    # Update default parameters with the custom ones passed as keyword arguments
    augmentation_parameters.update(**augmentation_params)
    return augmentation_parameters
