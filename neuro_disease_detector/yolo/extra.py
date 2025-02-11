import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from neuro_disease_detector.yolo.validation_consensus import stack_masks

def write_csv(metrics: dict, output_path: str) -> None:
    header = list(metrics[0].keys())
    with open(Path(output_path) / "metrics.csv" , mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for item in metrics:
            row = list(item.values())
            writer.writerow(row)
    return Path(output_path) / "metrics.csv"

def create_metrics_graphs(csv_path: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(15, 12)) 
    plt.suptitle("test graphs for 128 epochs", fontsize=20) 
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
