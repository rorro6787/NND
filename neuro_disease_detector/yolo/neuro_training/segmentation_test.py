from neuro_disease_detector.yolo.utils.utils_nifti import load_nifti_image_bgr
from neuro_disease_detector.yolo.utils.utils_nifti import load_nifti_image
from neuro_disease_detector.yolo.neuro_training.cross_validation import stack_masks
from neuro_disease_detector.yolo.neuro_training.__init__ import yolo_model_no_suffix
from ultralytics import YOLO
import numpy as np
import os
import time
from pathlib import Path
import cv2
from neuro_disease_detector.logger import get_logger
logger = get_logger(__name__)

import pandas as pd
import matplotlib.pyplot as plt

from neuro_disease_detector.neuro_training.__init__ import fold_to_patient

import nibabel as nib
import csv
import torch
from time import sleep




def yolo_3d_voting(volume: np.ndarray, predictions_xyz: tuple) -> np.ndarray: 
    tam_x, tam_y, tam_z, _ = volume.shape
    votes = np.zeros((tam_x, tam_y, tam_z))
    
    predictions_x, predictions_y, predictions_z = predictions_xyz
    
    for index, prediction_x in enumerate(predictions_x):
        masks = prediction_x.masks
        stack = stack_masks(masks, votes[index,:,:].shape)
        votes[index,:,:] = votes[index,:,:] + stack

    for index, prediction_y in enumerate(predictions_y):
        masks = prediction_y.masks
        stack = stack_masks(masks, votes[:,index,:].shape)
        votes[:,index,:] = votes[:,index,:] + stack
    
    for index, prediction_z in enumerate(predictions_z):
        masks = prediction_z.masks
        stack = stack_masks(masks, votes[:,:,index].shape)
        votes[:,:,index] = votes[:,:,index] + stack
    
    return votes

def yolo_3d_voting_planes(volume: np.ndarray, predictions_xyz: tuple) -> tuple: 
    tam_x, tam_y, tam_z, _ = volume.shape
    votes_saggital, votes_coronal, votes_axial = (np.zeros((tam_x, tam_y, tam_z)) for _ in range(3))

    predictions_x, predictions_y, predictions_z = predictions_xyz

    
    for index, prediction_x in enumerate(predictions_x):
        masks = prediction_x.masks
        stack = stack_masks(masks, votes_saggital[index,:,:].shape)
        votes_saggital[index,:,:] = votes_saggital[index,:,:] + stack

    for index, prediction_y in enumerate(predictions_y):
        masks = prediction_y.masks
        stack = stack_masks(masks, votes_saggital[:,index,:].shape)
        votes_coronal[:,index,:] = votes_coronal[:,index,:] + stack
    
    for index, prediction_z in enumerate(predictions_z):
        masks = prediction_z.masks
        stack = stack_masks(masks, votes_saggital[:,:,index].shape)
        votes_axial[:,:,index] = votes_axial[:,:,index] + stack
    
    return votes_saggital, votes_coronal, votes_axial

def yolo_3d_prediction(volume: np.ndarray, yolo_model: YOLO) -> np.ndarray:
    tam_x, tam_y, tam_z, _ = volume.shape
    
    slices_x = [volume[i,:,:] for i in range(tam_x)]
    slices_y = [volume[:,j,:] for j in range(tam_y)]
    slices_z = [volume[:,:,k] for k in range(tam_z)]

    predictions_x = _yolo_3d_prediction(yolo_model, slices_x)
    predictions_y = _yolo_3d_prediction(yolo_model, slices_y)
    predictions_z = _yolo_3d_prediction(yolo_model, slices_z)

    return predictions_x, predictions_y, predictions_z

def _yolo_3d_prediction(yolo_model:YOLO, slices: list, batch_size: int = 256):
    predictions = []
    with torch.no_grad():
        for i in range(0, len(slices), batch_size):
            batch = slices[i:i+batch_size]
            predictions_cuda = yolo_model(batch, save=False, verbose=False, show_boxes=False)
            predictions.extend(predictions_cuda)
            del predictions_cuda  
            torch.cuda.empty_cache()  
    return predictions



def test_neuro_system_k_folds(training_results_path: str, dataset_path: str, extra: bool = False) -> None: 
    """
    Test a YOLO-based neuro system across K-folds and generate performance metrics over time.

    This function iterates over a predefined set of folds, evaluates the performance 
    of different YOLO model checkpoints within each fold, and produces CSV files and 
    graphs summarizing the metrics for each fold.

    Args:
        training_results_path (str): 
            Path to the directory where the results of training (organized by folds) are stored. 
            Each fold should contain a `weights` subdirectory with YOLO model checkpoint files.
        
        dataset_path (str): 
            Path to the dataset being used for evaluation. This is required to test the neuro 
            system with the YOLO models.

    Workflow:
        1. Define the folds to process (fold1, fold2, ..., fold5).
        2. For each fold:
           a. Access the `weights` directory containing YOLO checkpoint files.
           b. Iterate over all checkpoint files (`*.pt`) excluding the `best.pt` file.
           c. Test the neuro system using the `test_neuro_system` function with the dataset, 
              current fold, and model checkpoint.
           d. Collect the metrics over time for the fold.
        3. Save the collected metrics for each fold to a CSV file using the `write_csv` function.
        4. Generate and save visualizations of the metrics using the `create_metrics_graphs` function.

    Note:
        - The `test_neuro_system` function is assumed to handle the evaluation of a single 
          model on the dataset for a given fold and return relevant metrics.
        - The `write_csv` and `create_metrics_graphs` functions handle the creation of CSV 
          files and graphs, respectively.

    Returns:
        None
    """

    # Define the folds to process
    folds = ["kfold-5"]

    # Iterate over each fold in the list
    for _, fold in enumerate(folds):
        # Define the path to the current fold within the training results directory
        fold_path = Path(training_results_path) / f"{yolo_model_no_suffix}-{fold}"

        # Initialize an empty list to store metrics over time for the current fold
        metrics_over_time = {}

        yolo_models_path = Path(fold_path) / "weights"

        # Iterate over all YOLO model checkpoint files in the weights directory
        for i in range(0, 81, 10):  
            # Construct the full path to the YOLO model checkpoint file
            yolo_model_path = Path(yolo_models_path) / f"epoch{i}.pt"    

            # Test the neuro system using the current fold and YOLO model checkpoint
            print(f"Starting validation of the Dataset with model {yolo_model_path}")

            metrics = test_neuro_system(dataset_path + "/MSLesSeg-Dataset/train", f"fold5", yolo_model_path, extra=extra)

            # Append the metrics to the list for the current fold
            metrics_over_time[i] = metrics


        # Save the collected metrics for the current fold to a CSV file
        write_csv(metrics_over_time, fold_path/"dsc_results_test.csv", extra=extra)

        # Generate and save visualizations of the metrics for the current fold
        # create_metrics_graphs(csv_path, training_results_path)


def write_csv(metrics: list, output_path: str, extra: bool = False) -> None:
    # Escribir en el archivo CSV

    if extra:
        headers = ["index", "dsc", "dscs", "dscc", "dsca"]

    else:
        headers = ["index", "dsc"]
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers) 
        
        for index, values in metrics.items():
            writer.writerow([index] + list(values)) 

    print(f"CSV file with DSC values over time created: {output_path}")

def test_neuro_system(dataset_path: str, fold: str, yolo_model_path: str, extra: bool = False) -> dict:
    
    confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    if extra:
        confusion_matrix_s = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        confusion_matrix_c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        confusion_matrix_a = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    yolo_model = YOLO(model=yolo_model_path, task="segment", verbose=False)
    
    for i in range(fold_to_patient[fold][0], fold_to_patient[fold][1]):
        pd = f'P{i}'
        patient_path = dataset_path + f"/{pd}"

        for timepoint_directory in Path(patient_path).iterdir():
            if not timepoint_directory.is_dir():
                continue
            
            td = timepoint_directory.name
            patient_timepoint_path = patient_path + f"/{td}"

            mask = patient_timepoint_path + f"/{pd}_{td}_MASK.nii"
            flair = patient_timepoint_path + f"/{pd}_{td}_FLAIR.nii"
            t1 = patient_timepoint_path + f"/{pd}_{td}_T1.nii"
            t2 = patient_timepoint_path + f"/{pd}_{td}_T2.nii"   

            # scan_types = [flair, t1, t2]
            scan_types = [flair]
            for scan in scan_types:
                print(f"The following scan will be tested: {Path(scan).name}")
                volume_yolo = load_nifti_image_bgr(scan)
                predictions_xyz = yolo_3d_prediction(volume_yolo, yolo_model)
                votes_scan = yolo_3d_voting(volume_yolo, predictions_xyz)
                
                if extra:
                    votes_saggital, votes_coronal, votes_axial = yolo_3d_voting_planes(volume_yolo, predictions_xyz)
                
                volume_mask = load_nifti_image(mask) 

                consensus_value = 2.0

                votes_scan_consensus = np.where(votes_scan >= consensus_value, 1.0, 0.0)

                confusion_matrix = update_confusion_matrix(confusion_matrix, votes_scan_consensus, volume_mask)
                if extra:
                    confusion_matrix_s = update_confusion_matrix(confusion_matrix_s, votes_saggital, volume_mask)
                    confusion_matrix_c = update_confusion_matrix(confusion_matrix_c, votes_coronal, volume_mask)
                    confusion_matrix_a = update_confusion_matrix(confusion_matrix_a, votes_axial, volume_mask)

                print(str(confusion_matrix) + " DSC consensus: " + str(compute_dsc(confusion_matrix)))
                print(str(confusion_matrix_s) + " DSC saggital: " + str(compute_dsc(confusion_matrix_s)))
                print(str(confusion_matrix_c) + " DSC coronal: " + str(compute_dsc(confusion_matrix_c)))
                print(str(confusion_matrix_a) + " DSC axial: " + str(compute_dsc(confusion_matrix_a)))
                sleep(3)

    cm_dsc = compute_dsc(confusion_matrix)

    if extra:
        cm_dsc_s = compute_dsc(confusion_matrix_s)
        cm_dsc_c = compute_dsc(confusion_matrix_c)
        cm_dsc_a = compute_dsc(confusion_matrix_a)
        return cm_dsc, cm_dsc_s, cm_dsc_c, cm_dsc_a
    
    return cm_dsc

def compute_dsc(confusion_matrix: dict) -> float:
    """
    Compute the Dice Similarity Coefficient (DSC) from a confusion matrix.

    Args:
        confusion_matrix (dict): 
            Dictionary containing the counts for TP, FP, TN, and FN.

    Returns:
        float: 
            Dice Similarity Coefficient (DSC) computed from the confusion matrix.
    """

    return (2*confusion_matrix["TP"]) / (2*confusion_matrix["TP"] + confusion_matrix["FP"] + confusion_matrix["FN"])

def update_confusion_matrix(confusion_matrix: dict, votes_scan: np.ndarray, mask_volume: np.ndarray) -> dict:
    """
    Update the confusion matrix based on the votes and the ground truth mask.

    Args:
        confusion_matrix (dict): 
            Dictionary containing the counts for TP, FP, TN, and FN.
        
        votes (np.ndarray): 
            Array of votes generated by the neuro system.
        
        mask (np.ndarray): 
            Ground truth mask for the volume.

    Returns:
        dict: 
            Updated confusion matrix with the counts for TP, FP, TN, and FN.
    """
    # Cálculo de TP, TN, FP, FN
    confusion_matrix["TP"] += np.sum((mask_volume == 1) & (votes_scan == 1))
    confusion_matrix["TN"] += np.sum((mask_volume == 0) & (votes_scan == 0))
    confusion_matrix["FN"] += np.sum((mask_volume == 1) & (votes_scan == 0))
    confusion_matrix["FP"] += np.sum((mask_volume == 0) & (votes_scan == 1))

    return confusion_matrix

def plot_results(csv_train_path: str, csv_test_path: str):
    # Load your CSV file
    data_train = pd.read_csv(csv_train_path)
    data_test = pd.read_csv(csv_test_path)

    plt.figure(figsize=(14, 8))

    plt.plot(data_test['index'], data_test['dsc'], label='test - CONSENSUS', marker='o', markersize=4, color='blue')
    plt.plot(data_test['index'], data_test['dscs'], label='test - SAGGITAL', marker='o', markersize=4, color='blue', linestyle='--')
    plt.plot(data_test['index'], data_test['dscc'], label='test - CORONAL', marker='o', markersize=4, color='blue', linestyle=':')
    plt.plot(data_test['index'], data_test['dsca'], label='test - AXIAL', marker='o', markersize=4, color='blue', linestyle='-.')

    # Plot the 'train' results with another color (e.g., red)
    plt.plot(data_train['index'], data_train['dsc'], label='train - CONSENSUS', marker='o', markersize=4, color='red')
    plt.plot(data_train['index'], data_train['dscs'], label='train - SAGGITAL', marker='o', markersize=4, color='red', linestyle='--')
    plt.plot(data_train['index'], data_train['dscc'], label='train - CORONAL', marker='o', markersize=4, color='red', linestyle=':')
    plt.plot(data_train['index'], data_train['dsca'], label='train - AXIAL', marker='o', markersize=4, color='red', linestyle='-.')

    # Add labels, title, and legend
    plt.xlabel('Batch')
    plt.ylabel('DCS')
    plt.title('Comparison of DCS Metric - Test vs Train')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f"{os.path.splitext(csv_test_path)[0]}.png")

    # Close the plot to avoid displaying it
    plt.close()



if __name__ == "__main__":
    yolo_model_path = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/neuro_training/runs"
    dataset_path = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/data_processing"

    test_neuro_system_k_folds(yolo_model_path, dataset_path, extra=True)
    train = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/neuro_training/runs/yolov8n-seg-kfold-5/dsc_results_train.csv"
    test = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/neuro_training/runs/yolov8n-seg-kfold-5/dsc_results_test.csv"
   
    # plot_results(train, test)
