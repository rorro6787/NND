import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
from pathlib import Path

def update_confusion_matrix(metrics: dict, real_mask: np.ndarray, predicted_mask: np.ndarray) -> dict:
    """
    Calculate pixel-level statistics for model evaluation.

    This function calculates True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) based on the ground truth and predicted segmentation masks.

    Parameters:
        metrics (dict): A dictionary containing pixel-level statistics (TP, FP, TN, FN).
        ground_truth_mask (np.ndarray): A binary mask representing the ground truth segmentation.
        predicted_mask (np.ndarray): A binary mask representing the predicted segmentation.
    
    Returns:
        dict: A dictionary containing updated pixel-level statistics after processing the current masks.
    
    Example:
        metrics = {"TP": 100, "FP": 20, "TN": 50, "FN": 10}
        ground_truth_mask = np.array([[0, 1], [1, 0]])
        predicted_mask = np.array([[0, 1], [0, 1]])
        metrics = calculate_metrics(metrics, ground_truth_mask, predicted_mask)
        print(metrics)
        # Output: {'TP': 101, 'FP': 21, 'TN': 49, 'FN': 11}
    """

    total_pixels = real_mask.size
    TP = np.sum(np.logical_and(real_mask, predicted_mask))  
    FP = np.sum(np.logical_and(np.logical_not(real_mask), predicted_mask))  
    TN = np.sum(np.logical_and(np.logical_not(real_mask), np.logical_not(predicted_mask))) 
    FN = total_pixels - (TP + FP + TN)  

    metrics["TP"] += TP
    metrics["FP"] += FP
    metrics["TN"] += TN
    metrics["FN"] += FN

    return metrics

def calculate_metrics(metrics: dict) -> dict:
    """
    Calculate evaluation metrics based on pixel-level statistics.

    This function computes various evaluation metrics (e.g., Recall, Precision, F1 Score)

    Parameters:
        metrics (dict): A dictionary containing pixel-level statistics (TP, FP, TN, FN).
    
    Returns:
        dict: A dictionary containing evaluation metrics calculated from the pixel-level statistics.
    
    Example:
        metrics = {"TP": 100, "FP": 20, "TN": 50, "FN": 10}
        formulas = calculate_formulas(metrics)
        print(formulas)
        # Output: {'Recall': 0.9090909090909091, 'Precision': 0.8333333333333334, 'Acc': 0.8333333333333334, 'Sensibility': 0.9090909090909091, 'IOU': 0.9090909090909091, 'F1 Score': 0.8695652173913043}
    """

    recall = metrics["TP"] / (metrics["TP"] + metrics["FN"])
    precision = metrics["TP"] / (metrics["TP"] + metrics["FP"])
    acc = (metrics["TP"] + metrics["TN"]) / (metrics["TP"] + metrics["FP"] + metrics["TN"] + metrics["FN"])
    sensibility = metrics["TP"] / (metrics["TP"] + metrics["FN"] + metrics["FP"])
    iou = metrics["TP"] / (metrics["TP"] + metrics["FN"] + metrics["FP"])
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return { 
        "Recall": np.nan_to_num(recall, nan=0),
        "Precision": np.nan_to_num(precision, nan=0), 
        "Acc": np.nan_to_num(acc, nan=0), 
        "Sensibility": np.nan_to_num(sensibility, nan=0), 
        "IOU": np.nan_to_num(iou, nan=0), 
        "F1 Score": np.nan_to_num(f1_score, nan=0)
    }

def write_csv(metrics: dict, output_path: str) -> None:
    """
    Writes performance metrics to a CSV file.

    Parameters:
    metrics (dict): A dictionary containing performance metrics. 
                    Each item in the dictionary is expected to be a tuple where:
                    - The first element is a dictionary with true positive (TP), 
                      false positive (FP), true negative (TN), false negative (FN), 
                      recall, precision, accuracy (Acc), sensibility, intersection over union (IOU), 
                      and F1 score.
                    - The second element is also a dictionary with similar metrics.

    path (str): The directory path where the output CSV file will be saved.

    Returns:
    None: This function does not return any value. It writes the output directly to a CSV file.

    CSV Format:
    The first row of the CSV file contains the headers:
    ['TP', 'FP', 'TN', 'FN', 'Recall', 'Precision', 'Acc', 'Sensibility', 'IOU', 'F1 Score']

    Each subsequent row contains the metrics for a specific item from the input dictionary,
    combining values from both dictionaries in the tuple.

    Note:
    The output CSV file is named 'output.csv' and is created in the specified directory path.
    
    Example:
    metrics = {
        ({"TP": 10, "FP": 2, "TN": 15, "FN": 1, "Recall": 0.91, "Precision": 0.83, "Acc": 0.92, "Sensibility": 0.89, "IOU": 0.75, "F1 Score": 0.87},
         {"TP": 8, "FP": 3, "TN": 12, "FN": 2, "Recall": 0.80, "Precision": 0.73, "Acc": 0.85, "Sensibility": 0.78, "IOU": 0.70, "F1 Score": 0.76}),
        # more items...
    }
    write_csv(metrics, '/path/to/directory')
    """

    # Extract dictionary keys to use as the header row for the CSV
    header = list(metrics[0].keys())

    # Open the file at the specified path in write mode
    with open(Path(output_path) / "metrics.csv" , mode='w', newline='') as file:
        # Initialize the CSV writer and write the header row
        writer = csv.writer(file)
        writer.writerow(header)
        
        # Iterate over the metrics dictionary and write rows to the file.
        for item in metrics:
            # Combine values from the dictionary.
            row = list(item.values())
            writer.writerow(row)

    return Path(output_path) / "metrics.csv"

def create_metrics_graphs(csv_path: str, output_path: str) -> None:
    """
    Generates line graphs for metrics stored in a CSV file, saving the output as a PNG image.

    Parameters:
    - csv_path (str): The path to the CSV file containing metric data. Each column should represent a metric.
    - test (bool): If True, the generated graphs will be labeled as 'Test'; otherwise, they will be labeled as 'Validation'.
    
    Output:
    - Saves a PNG file containing the graphs in the same directory as the input CSV file. 
      The file is named 'test.png' if test=True, otherwise 'val.png'.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Set up the figure for the graphs
    plt.figure(figsize=(15, 12))  # Create a figure with a size of 15x12 inches
    plt.suptitle("test graphs for 128 epochs", fontsize=20)  # Add a title

    for i, column in enumerate(df.columns):
        plt.subplot(3, 4, i + 1)
        plt.plot(df.index*3, df[column], marker='o')
        plt.title(column)
        plt.xlabel('Epoch')
        plt.ylabel(column)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(Path(output_path) / "metrics.png")  
    plt.close() 
