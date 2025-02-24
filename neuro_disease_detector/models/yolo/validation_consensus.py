
from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
import re

from neuro_disease_detector.models.yolo.__init__ import CM, Metrics
from neuro_disease_detector.utils.utils_dataset import get_timepoints_patient, get_patients_split
from neuro_disease_detector.models.yolo.utils.utils_nifti import load_nifti_image, load_nifti_image_bgr
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)

class YoloFoldValidator:
    def __init__(self, folds_directory: str, data_folder: str, consensus_threshold: int=1) -> None:
        self.folds_dir = folds_directory
        self.data_folder = data_folder
        self.cth = consensus_threshold
        self.k = 5

        self.cm_fold_epoch = {}
        self.metrics_fold_epoch = {}

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

    def validate_fold(self, fold: str) -> tuple:
        """Validates the model's performance for each epoch in the specified fold by processing all patients and storing confusion matrices and metrics."""

        # Construct the path to the current fold directory.
        fold_path = f"{self.folds_dir}/{fold}"
        
        # Define the path to the models directory inside the current fold.
        models_path = f"{fold_path}/weights"

        # Dictionaries to store confusion matrix and metrics for each epoch.
        cm_epoch = {}
        metrics_epoch = {}

        # Iterate through the files in the model weights directory.
        for file_name in os.listdir(models_path):
            # Construct the full file path for each model file.
            file_path = f"{models_path}/{file_name}"

            if (not os.path.isfile(file_path)) or (file_name not in ["best.pt"]):
                continue
            
            # epoch = int(re.search(r'\d+', file_name).group())

            # Create a YoloValidator object for the current model file and process the data.
            fold_split = f"fold{self.k - int(fold[-1]) + 1}"
            yolo_validator = YoloValidator(file_path, self.data_folder, self.cth, fold_split)
            yolo_validator.process_all_patients()

            # Store the confusion matrix and metrics for the current epoch.
            cm_epoch["best"] = yolo_validator.cm
            metrics_epoch["best"] = yolo_validator.metrics

        # Return the confusion matrix and metrics for all processed epochs.
        return cm_epoch, metrics_epoch
        
class YoloValidator:
    def __init__(self, model_path: str, data_folder: str, consensus_threshold: int, fold_split: str) -> None:
        self.model = YOLO(model_path, task="segmentation", verbose=False)
        self.data_path = f"{data_folder}/MSLesSeg-Dataset/train"
        self.test_patients = get_patients_split(fold_split)

        self.cm = { CM.TP : 0, CM.FP : 0, CM.TN : 0, CM.FN : 0}
        self.metrics = None
        self.cth = consensus_threshold

    def process_all_patients(self) -> None:
        """Processes all patients by iterating over each patient and calling _process_patient."""

        # Iterate over all patients specified by test_patients and process it.
        print(self.test_patients)
        for pd in range(self.test_patients[0], self.test_patients[1]):
            print(pd)
            self._process_patient(pd)

        # After processing all patients, compute the evaluation metrics.
        self.compute_metrics()

    def _process_patient(self, pd: int) -> None:
        """Processes a specific patient by loading scan data and running predictions for each timepoint."""

        # Get the number of timepoints available for this patient.
        num_tp = get_timepoints_patient(pd)
        pd_path = f"{self.data_path}/P{pd}"

        # Iterate over each timepoint for the current patient.
        for tp in range(1, num_tp+1):
            tp_path = f"{pd_path}/T{tp}"

            # Load the mask and the different scan types (FLAIR, T1, T2) for the current timepoint.
            mask = load_nifti_image(f"{tp_path}/P{pd}_T{tp}_MASK.nii")
            flair = load_nifti_image_bgr(f"{tp_path}/P{pd}_T{tp}_FLAIR.nii")
            t1 = load_nifti_image_bgr(f"{tp_path}/P{pd}_T{tp}_T1.nii")
            t2 = load_nifti_image_bgr(f"{tp_path}/P{pd}_T{tp}_T2.nii")
            scans = [flair, t1, t2]

            # Iterate through each scan to perform prediction and voting.
            for scan in scans:
                # Run the 3D YOLO model to get predictions for the current scan.
                predictions_xyz = yolo_3d_prediction(scan, self.model)

                # Generate votes based on the predictions.
                votes = yolo_3d_votes(scan.shape, predictions_xyz)

                # Apply consensus to the votes and get a final decision.
                votes_consensus = self.apply_consensus(votes)

                # Update the confusion matrix with the final consensus votes and the ground truth mask
                self.update_cm(votes_consensus, mask)
            
    def apply_consensus(self, votes: np.ndarray) -> np.ndarray:
        """Applies consensus threshold to the votes array and returns a binary result."""
        return np.where(votes >= self.cth, 1.0, 0.0)
    
    def compute_metrics(self) -> None:
        """Computes various performance metrics (Recall, Precision, Accuracy, Sensibility, IoU, DSC, F1) from the confusion matrix."""

        # Compute recall: higher is better
        recall = self.cm[CM.TP] / (self.cm[CM.TP] + self.cm[CM.FN])

        # Compute precision: higher is better
        precision = self.cm[CM.TP] / (self.cm[CM.TP] + self.cm[CM.FP])

        # Compute accuracy: higher is better
        acc = (self.cm[CM.TP] + self.cm[CM.TN]) / (self.cm[CM.TP] + self.cm[CM.FP] + self.cm[CM.TN] + self.cm[CM.FN])

        # Compute sensibility: higher is better
        sensibility = self.cm[CM.TP] / (self.cm[CM.TP] + self.cm[CM.FN] + self.cm[CM.FP])

        # Compute Intersection over Union (IoU): higher is better
        iou = self.cm[CM.TP] / (self.cm[CM.TP] + self.cm[CM.FN] + self.cm[CM.FP])

        # Compute Dice Similarity Coefficient (DSC): higher is better
        dsc = 2*self.cm[CM.TP] / (2*self.cm[CM.TP] + self.cm[CM.FP] + self.cm[CM.FN])
        
        # Return a dictionary of all the computed metrics with nan values replaced by 0
        self.metrics = { 
            Metrics.RECALL : np.nan_to_num(recall, nan=0),
            Metrics.PRECISION : np.nan_to_num(precision, nan=0), 
            Metrics.ACCUARICY : np.nan_to_num(acc, nan=0), 
            Metrics.SENSIBILITY : np.nan_to_num(sensibility, nan=0), 
            Metrics.IOU : np.nan_to_num(iou, nan=0), 
            Metrics.DSC : np.nan_to_num(dsc, nan=0),
        }

    def update_cm(self, prediction: np.ndarray, mask: np.ndarray) -> None:
        """Update the confusion matrix based on the votes and the ground truth mask."""

        # Computation of TP, TN, FN, FP
        self.cm[CM.TP] += np.sum((mask == 1) & (prediction == 1))
        self.cm[CM.TN] += np.sum((mask == 0) & (prediction == 0))
        self.cm[CM.FN] += np.sum((mask == 1) & (prediction == 0))
        self.cm[CM.FP] += np.sum((mask == 0) & (prediction == 1))

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
