
from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import re
import json

from neuro_disease_detector.models.yolo.__init__ import CM, Metrics, Validator
from neuro_disease_detector.utils.utils_dataset import get_timepoints_patient, get_patients_split
from neuro_disease_detector.models.yolo.utils.utils_nifti import load_nifti_image, load_nifti_image_bgr
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)

class YoloFoldValidator:
    def __init__(self, folds_directory: str, validator: Validator, data_folder: str, consensus_threshold: int = 2) -> None:
        self.folds_dir = folds_directory
        self.validator = validator
        self.data_folder = data_folder
        self.cth = consensus_threshold

        self.k = 5

    def validate_all_folds(self) -> None:
        """Validates the model's performance for all folds and stores confusion matrices and metrics for each fold."""

        # Iterate over all folds (from fold1 to foldk).
        for i in range(self.k):
            # Define the fold name dynamically (fold1, fold2, ..., foldk).
            fold = f"fold{i+1}"

            # Validate the current fold and retrieve the confusion matrix and metrics for each epoch.
            cm_epoch, metrics_epoch = self.validate_fold(fold)

            # Store the results (confusion matrix and metrics) for the current fold.
            self.cm_fold_epoch[fold] = cm_epoch
            self.metrics_fold_epoch[fold] = metrics_epoch

    def validate_fold(self, fold: str, test: bool = False) -> tuple:
        """Validates the model's performance for each epoch in the specified fold by processing all patients and storing confusion matrices and metrics."""

        # Construct the path to the current fold directory.
        fold_path = f"{self.folds_dir}/{fold}"
        
        # Define the path to the models directory inside the current fold.
        model_path = f"{fold_path}/weights/best.pt"

        # Create a YoloValidator object for the current model file and process the data.
        if test:
            fold_split = "test"
        else:
            fold_split = f"fold{self.k - int(fold[-1]) + 1}"

        json_path = f"{self.folds_dir}/summary{fold[-1]}.json"
        yolo_validator = YoloValidator(model_path, self.validator, self.data_folder, self.cth, fold_split, json_path)
        yolo_validator.process_all_patients()
        
class YoloValidator:
    def __init__(self, model_path: str, validator: Validator, data_folder: str, cth: int, fold_split: str, json_path: str) -> None:
        self.model = YOLO(model_path, task="segmentation", verbose=False)
        self.validator = validator
        self.data_path = f"{data_folder}/MSLesSeg-Dataset/train"
        self.cth = cth
        self.fold_split = fold_split

        self.json_path = json_path
        self.cm = { CM.TP : 0, CM.FP : 0, CM.TN : 0, CM.FN : 0}        

    def process_all_patients(self) -> None:
        """Processes all patients by iterating over each patient and calling _process_patient."""

        test_patients = get_patients_split(self.fold_split)

        # Iterate over all patients in the test set
        for pd in range(test_patients[0], test_patients[1]):
            # Define the path for the current patient's data
            pd_path = f"{self.data_path}/P{pd}"

            # Loop over all timepoints for the current patient
            for tp in range(1, get_timepoints_patient(pd)+1):
                # Define the path for the current timepoint data
                tp_path = f"{pd_path}/T{tp}"

                # Process the patient's data for the given timepoint and update the confusion matrix
                cm = self._process_patient_timepoint(pd, tp, tp_path)
                metrics = compute_metrics(cm)
                self.update_cm(cm)

                if self.fold_split == "test":
                    self.dump_json(f"P{pd}T{tp}", cm, metrics)

        # After processing all patients, compute the evaluation metrics.
        metrics = compute_metrics(self.cm)

        # If the fold split is "test", dump the final results for the test set
        if self.fold_split == "test":
            self.dump_json("Test", self.cm, metrics)
        else:
            # Otherwise, dump the final results for the validation set
            self.dump_json("Val", self.cm, metrics)

    def dump_json(self, index: str, cm: dict, metrics: dict) -> None:
        """Dumps the confusion matrix and metrics to a JSON file."""

        # Load the existing JSON file if it exists, or create a new dictionary.
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as file:
                data = json.load(file)
        else:
            data = {}
        
        # Update the data dictionary with the new confusion matrix and metrics.
        data[index] = { **cm, **metrics }

        # Dump the updated data dictionary to the JSON file with indentation.
        with open(self.json_path, "w") as file:
            json.dump(data, file, indent=4)

    def _process_patient_timepoint(self, pd: int, tp: int, tp_path: str) -> None:
        """Processes a specific patient by loading scan data and running predictions for each timepoint."""

        # Load the mask and the different scan types (FLAIR, T1, T2) for the current timepoint.
        mask = load_nifti_image(f"{tp_path}/P{pd}_T{tp}_MASK.nii")
        flair = load_nifti_image_bgr(f"{tp_path}/P{pd}_T{tp}_FLAIR.nii")
        t1 = load_nifti_image_bgr(f"{tp_path}/P{pd}_T{tp}_T1.nii")
        t2 = load_nifti_image_bgr(f"{tp_path}/P{pd}_T{tp}_T2.nii")
        scans = [flair, t1, t2]

        # Iterate through each scan to perform prediction and voting.
        for scan in scans:
            if self.validator == Validator.CONS2D or self.validator == Validator.CONS3D:
                predictions = yolo_3d_prediction(scan, self.model)
                votes = yolo_3d_votes(scan.shape, predictions)
            else:
                predictions = yolo_3d_prediction_plane(scan, self.model, self.validator)
                votes = yolo_3d_votes_plane(scan.shape, predictions, self.validator)
                
            # Apply consensus to the votes and get a final decision.
            votes_consensus = self.apply_consensus(votes)

            # Update the confusion matrix with the final consensus votes and the ground truth mask
            self.update_cm(votes_consensus, mask)
            cm = compute_cm(votes_consensus, mask)
            return cm
            
    def apply_consensus(self, votes: np.ndarray) -> np.ndarray:
        """Applies consensus threshold to the votes array and returns a binary result."""
        if self.validator == Validator.CONS2D or self.validator == Validator.CONS3D:
            return np.where(votes >= self.cth, 1.0, 0.0)
        return votes

    def update_cm(self, cm: dict) -> None:
        """Update the confusion matrix based on the votes and the ground truth mask."""

        # Computation of TP, TN, FN, FP
        self.cm[CM.TP] += cm[CM.TP]
        self.cm[CM.TN] += cm[CM.TN]
        self.cm[CM.FN] += cm[CM.FN]
        self.cm[CM.FP] += cm[CM.FP]

def compute_cm(prediction: np.ndarray, mask: np.ndarray) -> dict:
    """Computes a confusion matrix (TP, TN, FP, FN) from a binary prediction and ground truth mask."""

    # Initialize the confusion matrix with zeros.
    cm = { CM.TP : 0, CM.TN : 0, CM.FP : 0, CM.FN : 0 }

    # Computation of TP, TN, FN, FP
    cm[CM.TP] = np.sum((mask == 1) & (prediction == 1))
    cm[CM.TN] = np.sum((mask == 0) & (prediction == 0))
    cm[CM.FN] = np.sum((mask == 1) & (prediction == 0))
    cm[CM.FP] = np.sum((mask == 0) & (prediction == 1))

    return cm

def compute_metrics(cm: dict) -> dict:
    """Computes various performance metrics (Recall, Precision, Accuracy, Sensibility, IoU, DSC, F1) from the confusion matrix."""

    # Compute recall: higher is better
    recall = cm[CM.TP] / (cm[CM.TP] + cm[CM.FN])

    # Compute precision: higher is better
    precision = cm[CM.TP] / (cm[CM.TP] + cm[CM.FP])

    # Compute accuracy: higher is better
    acc = (cm[CM.TP] + cm[CM.TN]) / (cm[CM.TP] + cm[CM.FP] + cm[CM.TN] + cm[CM.FN])

    # Compute sensibility: higher is better
    sensibility = cm[CM.TP] / (cm[CM.TP] + cm[CM.FN] + cm[CM.FP])

    # Compute Intersection over Union (IoU): higher is better
    iou = cm[CM.TP] / (cm[CM.TP] + cm[CM.FN] + cm[CM.FP])

    # Compute Dice Similarity Coefficient (DSC): higher is better
    dsc = 2*cm[CM.TP] / (2*cm[CM.TP] + cm[CM.FP] + cm[CM.FN])
    
    # Return a dictionary of all the computed metrics with nan values replaced by 0
    return { 
        Metrics.RECALL : np.nan_to_num(recall, nan=0),
        Metrics.PRECISION : np.nan_to_num(precision, nan=0), 
        Metrics.ACCUARICY : np.nan_to_num(acc, nan=0), 
        Metrics.SENSIBILITY : np.nan_to_num(sensibility, nan=0), 
        Metrics.IOU : np.nan_to_num(iou, nan=0), 
        Metrics.DSC : np.nan_to_num(dsc, nan=0),
    }

def stack_masks(masks: list, image_shape: tuple) -> np.ndarray:
    """
    Combine a list of binary masks into a single mask by stacking and resizing them.

    Args:
        masks (list): A list of binary masks to combine.
        image_shape (tuple): The target shape (height, width) for the combined mask.

    Returns:
        np.ndarray: A single combined binary mask with the specified shape.
    """

    # Return an empty mask if the input list is empty.
    if not masks:
        return np.zeros((image_shape[0], image_shape[1]))

    # Convert masks to a NumPy array on the CPU for processing.
    masks = masks.data.cpu().numpy()

    # Resize the first mask to the target image shape and initialize the stack.
    stack = cv2.resize(masks[0], (image_shape[1], image_shape[0]))
    
    # Iterate through the remaining masks.
    for mask in masks[1:]:
        # Resize the current mask to the target image shape.
        resized_mask = cv2.resize(mask, (image_shape[1], image_shape[0]))

        # Combine the current mask with the stack using a logical OR operation.
        stack = np.logical_or(stack, resized_mask)
    
    # Convert the combined stack to an unsigned 8-bit integer array and return it.
    return stack.astype(np.uint8)

def yolo_3d_votes_plane(scan_shape: np.ndarray, predictions_xyz: tuple, validator: Validator) -> np.ndarray:
    """
    Accumulates YOLO-style 3D prediction masks into a 3D scan grid along the selected axis.

    Args:
        scan_shape (np.ndarray): Shape of the 3D scan grid (depth, height, width).
        predictions_x (np.ndarray): A 3D array with predictions along the selected axis.

    Returns:
        np.ndarray: A 3D array (same shape as scan_shape) with accumulated votes along the selectedaxis.
    """

    # Initialize a 3D array of zeros with the same shape as the input scan shape
    votes = np.zeros((scan_shape[0], scan_shape[1], scan_shape[2]))

    # Unpack the 3D predictions into individual x, y, and z components
    predictions_x, predictions_y, predictions_z = predictions_xyz

    if validator == Validator.SAGITTAL:
        for index, prediction_x in enumerate(predictions_x):
            # Generate a 3D mask stack from the prediction and update the votes over the x axis
            stack = stack_masks(prediction_x.masks, votes[index,:,:].shape)
            votes[index,:,:] = votes[index,:,:] + stack
    elif validator == Validator.CORONAL:
        for index, prediction_y in enumerate(predictions_y):
            # Generate a 3D mask stack from the prediction and update the votes over the y axis
            stack = stack_masks(prediction_y.masks, votes[:,index,:].shape)
            votes[:,index,:] = votes[:,index,:] + stack
    else:
        for index, prediction_z in enumerate(predictions_z):
            # Generate a 3D mask stack from the prediction and update the votes over the z axis
            stack = stack_masks(prediction_z.masks, votes[:,:,index].shape)
            votes[:,:,index] = votes[:,:,index] + stack
    
    # Return the final votes array
    return votes

def yolo_3d_votes(scan_shape: np.ndarray, predictions_xyz: tuple) -> np.ndarray: 
    """
    Accumulates YOLO-style 3D prediction masks into a 3D scan grid.

    Args:
        scan_shape (np.ndarray): Shape of the 3D scan grid (depth, height, width).
        predictions_xyz (tuple): A tuple containing three lists of predictions (x, y, z)

    Returns:
        np.ndarray: A 3D array (same shape as scan_shape) with accumulated votes
    """

    # Initialize a 3D array of zeros with the same shape as the input scan shape
    votes = np.zeros((scan_shape[0], scan_shape[1], scan_shape[2]))

    # Unpack the 3D predictions into individual x, y, and z components
    predictions_x, predictions_y, predictions_z = predictions_xyz
    
    for index, prediction_x in enumerate(predictions_x):
        # Generate a 3D mask stack from the prediction and update the votes over the x axis
        stack = stack_masks(prediction_x.masks, votes[index,:,:].shape)
        votes[index,:,:] = votes[index,:,:] + stack

    for index, prediction_y in enumerate(predictions_y):
        # Generate a 3D mask stack from the prediction and update the votes over the y axis
        stack = stack_masks(prediction_y.masks, votes[:,index,:].shape)
        votes[:,index,:] = votes[:,index,:] + stack
    
    for index, prediction_z in enumerate(predictions_z):
        # Generate a 3D mask stack from the prediction and update the votes over the z axis
        stack = stack_masks(prediction_z.masks, votes[:,:,index].shape)
        votes[:,:,index] = votes[:,:,index] + stack
    
    # Return the final votes array
    return votes

def yolo_3d_prediction_plane(volume: np.ndarray, yolo_model: YOLO, validator: Validator) -> np.ndarray:
    """
    Perform 3D predictions on a volume using a YOLO model by processing slices along the selected axis.

    Args:
        volume (np.ndarray): The 4D input volume with shape (tam_x, tam_y, tam_z, channels).
        yolo_model (YOLO): The YOLO model instance used for predictions.

    Returns:
        np.ndarray: Predictions for slices along the selected axis.
    """

    # Extract dimensions of the input volume.
    tam_x, tam_y, tam_z, _ = volume.shape
    
    if validator == Validator.SAGITTAL:
        # Create 2D slices along the x-axis.
        slices = [volume[i,:,:] for i in range(tam_x)]
    elif validator == Validator.CORONAL:
        # Create 2D slices along the y-axis.
        slices = [volume[:,i,:] for i in range(tam_y)]
    else:
        # Create 2D slices along the z-axis.
        slices = [volume[:,:,i] for i in range(tam_z)]

    # Perform predictions on slices along the x-axis using the YOLO model.
    predictions = _yolo_3d_prediction(yolo_model, slices)

    # Return the predictions for the x-axis.
    return predictions

def yolo_3d_prediction(volume: np.ndarray, yolo_model: YOLO) -> np.ndarray:
    """
    Perform 3D predictions on a volume using a YOLO model by processing slices along each axis.

    Args:
        volume (np.ndarray): The 4D input volume with shape (tam_x, tam_y, tam_z, channels).
        yolo_model (YOLO): The YOLO model instance used for predictions.

    Returns:
        tuple: A tuple containing predictions for slices along the x, y, and z axes.
    """

    # Extract dimensions of the input volume.
    tam_x, tam_y, tam_z, _ = volume.shape
    
    # Create 2D slices along each axis (x, y, z).
    slices_x = [volume[i,:,:] for i in range(tam_x)]
    slices_y = [volume[:,j,:] for j in range(tam_y)]
    slices_z = [volume[:,:,k] for k in range(tam_z)]

    # Perform predictions on slices along each axis using the YOLO model.
    predictions_x = _yolo_3d_prediction(yolo_model, slices_x)
    predictions_y = _yolo_3d_prediction(yolo_model, slices_y)
    predictions_z = _yolo_3d_prediction(yolo_model, slices_z)

    # Return the predictions for all three axes as a tuple.
    return predictions_x, predictions_y, predictions_z

def _yolo_3d_prediction(yolo_model: YOLO, slices: list, batch_size: int = 128) -> list:
    """
    Perform 3D predictions using a YOLO model on a list of 2D image slices.

    Args:
        yolo_model (YOLO): The YOLO model instance to use for predictions.
        slices (list): A list of 2D slices (e.g., image arrays) to process.
        batch_size (int): Number of slices to process at once in a batch. Default is 256.

    Returns:
        list: A list of prediction results for all slices.
    """
    predictions = []  # Initialize an empty list to store the predictions.

    # Disable gradient calculations to save memory and improve performance during inference.
    with torch.no_grad():
        # Process the slices in batches.
        for i in range(0, len(slices), batch_size):
            # Extract the current batch of slices.
            batch = slices[i:i + batch_size]

            # Perform predictions on the current batch using the YOLO model.
            predictions_cuda = yolo_model(batch, save=False, verbose=False, show_boxes=False)

            # Extend the predictions list with results from the current batch.
            predictions.extend(predictions_cuda)

            # Delete the temporary CUDA predictions to free up GPU memory.
            del predictions_cuda

            # Clear the CUDA cache to avoid potential memory issues.
            torch.cuda.empty_cache()

    # Return the final list of predictions for all slices.
    return predictions
