from neuro_disease_detector.utils.utils_dataset import load_nifti_image_bgr
from neuro_disease_detector.utils.utils_dataset import load_nifti_image
from neuro_disease_detector.neuro_training.cross_validation import stack_masks
from ultralytics import YOLO
import numpy as np
import os
import time
from pathlib import Path
import cv2

import nibabel as nib


def test_neuro_system(dataset_path: str, yolo_model_path: str) -> dict:
    # Construct the paths to the images and masks directories for the specified fold
    fold_path = os.path.join(dataset_path, f'MSLesSeg-Dataset/train')

    # Initialize an empty dictionary for the test batch and a batch size of 128
    batch = {}
    batch_size = 128

    # Initialize the confusion matrix with zero counts for TP, FP, TN, FN
    confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    
    for i in range(6):
        print(i)
        pd = f'P{i+1}'
        fold_path_aux = os.path.join(fold_path, f'P{i+1}')

        for timepoint_directory in Path(fold_path_aux).iterdir():
            if not timepoint_directory.is_dir():
                continue
            
            td = timepoint_directory.name
            fold_path_aux2 = os.path.join(fold_path_aux, td)
            mask = os.path.join(fold_path_aux2, f'{pd}_{td}_MASK.nii')
            flair = os.path.join(fold_path_aux2, f'{pd}_{td}_FLAIR.nii')
            t1 = os.path.join(fold_path_aux2, f'{pd}_{td}_T1.nii')
            t2 = os.path.join(fold_path_aux2, f'{pd}_{td}_T2.nii')    


            volume = load_nifti_image(mask) 
            votes_flair = consensus(flair, yolo_model_path)      
            votes_t1 = consensus(t1, yolo_model_path)
            votes_t2 = consensus(t2, yolo_model_path)

            consenso = 4.0

            votes = np.where(votes_flair >= consenso, 1.0, 0.0)

            # Cálculo de TP, TN, FP, FN
            confusion_matrix["TP"] += np.sum((volume == 1) & (votes == 1))
            confusion_matrix["TN"] += np.sum((volume == 0) & (votes == 0))
            confusion_matrix["FP"] += np.sum((volume == 1) & (votes == 0))
            confusion_matrix["FN"] += np.sum((volume == 0) & (votes == 1))

            votes = np.where(votes_t1 >= consenso, 1.0, 0.0)

            confusion_matrix["TP"] += np.sum((volume == 1) & (votes == 1))
            confusion_matrix["TN"] += np.sum((volume == 0) & (votes == 0))
            confusion_matrix["FP"] += np.sum((volume == 1) & (votes == 0))
            confusion_matrix["FN"] += np.sum((volume == 0) & (votes == 1))


            votes = np.where(votes_t2 >= consenso, 1.0, 0.0)

            confusion_matrix["TP"] += np.sum((volume == 1) & (votes == 1))
            confusion_matrix["TN"] += np.sum((volume == 0) & (votes == 0))
            confusion_matrix["FP"] += np.sum((volume == 1) & (votes == 0))
            confusion_matrix["FN"] += np.sum((volume == 0) & (votes == 1))

    return confusion_matrix
    
def consensus(file_path: str, yolo_model_path: str) -> None: 
    volume = load_nifti_image_bgr(file_path)
    tam_x, tam_y, tam_z, _ = volume.shape
    vv, vvs, vvc, vva = np.zeros((tam_x, tam_y, tam_z)), np.zeros((tam_x, tam_y, tam_z)), np.zeros((tam_x, tam_y, tam_z)), np.zeros((tam_x, tam_y, tam_z))
    
    slices_x = [volume[i,:,:] for i in range(tam_x)]
    slices_y = [volume[:,j,:] for j in range(tam_y)]
    slices_z = [volume[:,:,k] for k in range(tam_z)]

    model = YOLO(model=yolo_model_path, task="segment", verbose=False)
    predictions_x = model(slices_x, save=False, verbose=False, show_boxes=False)
    predictions_y = model(slices_y, save=False, verbose=False, show_boxes=False)
    predictions_z = model(slices_z, save=False, verbose=False, show_boxes=False)
    
    for index, prediction_x in enumerate(predictions_x):
        masks = prediction_x.masks
        stack = stack_masks(masks, vv[index,:,:].shape)
        vv[index,:,:] = vv[index,:,:] + stack
        vvs[index,:,:] = vvs[index,:,:] + stack

    for index, prediction_y in enumerate(predictions_y):
        masks = prediction_y.masks
        stack = stack_masks(masks, vv[:,index,:].shape)
        vv[:,index,:] = vv[:,index,:] + stack
        vvc[:,index,:] = vvc[:,index,:] + stack
    
    for index, prediction_z in enumerate(predictions_z):
        masks = prediction_z.masks
        stack = stack_masks(masks, vv[:,:,index].shape)
        vv[:,:,index] = vv[:,:,index] + stack
        vva[:,:,index] = vva[:,:,index] + stack
    
    return vv, vvs, vvc, vva




if __name__ == "__main__":
    
    yolo_model_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training/runs/yolov8n-seg-me-kfold-5/weights/best.pt"
    dataset_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training/MSLesSeg-Dataset/train/P6/T1/P6_T1_FLAIR.nii"
    mask_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training/MSLesSeg-Dataset/train/P6/T1/P6_T1_MASK.nii"
    vv, vvs, vvc, vva = consensus(dataset_path, yolo_model_path)
    consenso = 2.0

    votes = np.where(vv >= consenso, 1.0, 0.0)
    volume = load_nifti_image(dataset_path)
    mask = load_nifti_image(mask_path)
    print("Consenso: 2")
    print("Min votos: ", np.min(votes))
    print("Max votos: ", np.max(votes))
    print(votes.shape)


    # Crear una imagen NIfTI sin matriz de transformación (identidad por defecto)
    votes = nib.Nifti1Image(votes, affine=np.eye(4))  
    mask = nib.Nifti1Image(mask, affine=np.eye(4))
    volume = nib.Nifti1Image(volume, affine=np.eye(4))
    vvs = nib.Nifti1Image(vvs, affine=np.eye(4))
    vvc = nib.Nifti1Image(vvc, affine=np.eye(4))
    vva = nib.Nifti1Image(vva, affine=np.eye(4))

    # Guardar el archivo
    nib.save(volume, "pruebas_3D/P6_T1_FLAIR/original.nii")
    nib.save(mask, "pruebas_3D/P6_T1_FLAIR/mask.nii")
    nib.save(votes, "pruebas_3D/P6_T1_FLAIR/model_consenso.nii")
    nib.save(vvs, "pruebas_3D/P6_T1_FLAIR/model_sagital.nii")
    nib.save(vvc, "pruebas_3D/P6_T1_FLAIR/model_coronal.nii")
    nib.save(vva, "pruebas_3D/P6_T1_FLAIR/model_axial.nii")