import os
import numpy as np
import json
from sklearn.model_selection import KFold
from pathlib import Path
from neuro_disease_detector.nnu_net.__init__ import fold_to_patient
from neuro_disease_detector.nnu_net.__init__ import split_assign
import shutil

def create_nnu_dataset(dataset_dir: str):
    nnUNet_datapath = f"{os.getcwd()}/nnUNet_raw/Dataset024_MSLesSeg"
    os.makedirs(nnUNet_datapath, exist_ok=True)
    os.makedirs(f"{nnUNet_datapath}/imagesTr")
    os.makedirs(f"{nnUNet_datapath}/imagesTs")
    os.makedirs(f"{nnUNet_datapath}/labelsTr")
    
    dataset_path = f"{dataset_dir}/MSLesSeg-Dataset/train"
    for pd in range(1, 54):
        if pd == 30:
            continue
        pd_path = f"{dataset_path}/P{pd}"
        
        for td in range(1, 5):
            td_path = f"{pd_path}/T{td}"
            if not os.path.exists(td_path):
                break
            
            flair_path = f"{td_path}/P{pd}_T{td}_FLAIR.nii"
            t1_path = f"{td_path}/P{pd}_T{td}_T1.nii"
            t2_path = f"{td_path}/P{pd}_T{td}_T2.nii"
            mask_path = f"{td_path}/P{pd}_T{td}_MASK.nii"

            fold = split_assign(pd)

            if fold == "test":
                shutil.copy(flair_path, f"{nnUNet_datapath}/imagesTs/BRATS_{pd}_{td}_0000.nii.gz")
                shutil.copy(t1_path, f"{nnUNet_datapath}/imagesTs/BRATS_{pd}_{td}_0001.nii.gz")
                shutil.copy(t2_path, f"{nnUNet_datapath}/imagesTs/BRATS_{pd}_{td}_0002.nii.gz")
            else:
                shutil.copy(flair_path, f"{nnUNet_datapath}/imagesTr/BRATS_{pd}_{td}_0000.nii.gz")
                shutil.copy(t1_path, f"{nnUNet_datapath}/imagesTr/BRATS_{pd}_{td}_0001.nii.gz")
                shutil.copy(t2_path, f"{nnUNet_datapath}/imagesTr/BRATS_{pd}_{td}_0002.nii.gz")
                shutil.copy(mask_path, f"{nnUNet_datapath}/labelsTr/BRATS_{pd}_{td}.nii.gz")

            



                
            


dataset_path = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/nnu_net"
create_nnu_dataset(dataset_path)