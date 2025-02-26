from enum import Enum

class YoloModel(Enum):
    V11N_SEG = "yolo11n-seg.pt"
    V11S_SEG = "yolo11s-seg.pt"
    V11M_SEG = "yolo11m-seg.pt"
    V11L_SEG = "yolo11l-seg.pt"
    V11X_SEG = "yolo11x-seg.pt"

class Trainer(Enum):
    SIMPLE_AXIAL = "axial"
    SIMPLE_CORONAL = "coronal"
    SIMPLE_SAGITTAL = "sagittal"
    FULL_3D = "3d_fullres"

class Validator(Enum):
    CONS3D = "consensus3d"
    CONS2D = "consensus2d"
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"

class Metrics(Enum):
    RECALL = "Recall"
    PRECISION = "Precision"
    ACCUARICY = "Acc"
    SENSIBILITY = "Sensibility"
    IOU = "IOU"
    DSC = "DSC"

class CM(Enum):
    TP = "TP"
    TN = "TN"
    FP = "FP"
    FN = "FN"
    