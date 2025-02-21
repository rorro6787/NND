from enum import Enum

class Trainer(Enum):
    EPOCHS_1 = "nnUNetTrainer_1epoch"
    EPOCHS_5 = "nnUNetTrainer_5epochs"
    EPOCHS_10 = "nnUNetTrainer_10epochs"
    EPOCHS_20 = "nnUNetTrainer_20epochs"
    EPOCHS_50 = "nnUNetTrainer_50epochs"
    EPOCHS_100 = "nnUNetTrainer_100epochs"
    EPOCHS_250 = "nnUNetTrainer_250epochs"
    EPOCHS_500 = "nnUNetTrainer_500epochs"
    EPOCHS_750 = "nnUNetTrainer_750epochs"
    EPOCHS_2000 = "nnUNetTrainer_2000epochs"
    EPOCHS_4000 = "nnUNetTrainer_4000epochs"
    EPOCHS_8000 = "nnUNetTrainer_8000epochs"

class Fold(Enum):
    FOLD_1 = "0"
    FOLD_2 = "1"
    FOLD_3 = "2"
    FOLD_4 = "3"
    FOLD_5 = "4"

class Configuration(Enum):
    SIMPLE_2D = "2d"
    FULL_3D = "3d_fullres"
