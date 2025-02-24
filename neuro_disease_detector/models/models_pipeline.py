from neuro_disease_detector.models.nnUNet.__init__ import Configuration, Fold, Trainer
from neuro_disease_detector.models.nnUNet.nnUNet_pipeline import nnUNet

import pandas as pd
import os

if __name__ == "__main__":
    trainer = Trainer.EPOCHS_50
    csv_path = f"{os.getcwd()}/model_study.csv"
    for i in range(6):
        dataset_id = f"00{i}"     
        for configuration in Configuration:
            for fold in Fold:
                nnUNet(dataset_id, configuration, fold, trainer).execute_pipeline(csv_path)   
    

    
