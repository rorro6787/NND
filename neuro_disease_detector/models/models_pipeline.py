from neuro_disease_detector.models.nnUNet.nnUNet_pipeline import nnUNet
from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.models.yolo.yolo_pipeline import yolo_init

import os
import pandas as pd


def mean(csv_path: str):
    
    # Cargar el CSV
    data = pd.read_csv(csv_path)
    data = data[data['ValTest'] == 'test']
    data = data[data['model'] == 'nnUNet3D']
    # Agrupar por 'dataId', 'model', y 'ValTest', y calcular la media de cada métrica
    result = data.groupby(['dataId', 'model', 'ValTest']).agg({
        'dsc': 'mean',
        'iou': 'mean',
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean'
    }).reset_index()

    # Mostrar el resultado
    print(result)

def std(csv_path):
    # Cargar el CSV
    data = pd.read_csv(csv_path)
    data = data[data['ValTest'] == 'test']
    data = data[data['model'] == 'nnUNet3D']
    result = data.groupby(['dataId', 'model', 'ValTest']).agg({
        'dsc': 'std',
        'iou': 'std',
        'accuracy': 'std',
        'precision': 'std',
        'recall': 'std',
        'f1_score': 'std'
    }).reset_index()

    # Display the result
    print(result)


if __name__ == "__main__":
    dataset_id = "024"
    trainer = Trainer.EPOCHS_100
    csv_path = f"{os.getcwd()}/metrics.csv"
    """
    for configuration in Configuration:
        for fold in Fold:
            nnUNet(dataset_id, configuration, fold, trainer).init(csv_path)   
    """

    mean(csv_path)
    std(csv_path)

