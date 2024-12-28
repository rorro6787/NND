from neuro_disease_detector.utils.utils_dataset import load_nifti_image_bgr
from neuro_disease_detector.neuro_training.cross_validation import stack_masks
from ultralytics import YOLO
import numpy as np
import time

def consensus(file_path: str, yolo_model_path: str) -> None: 
    volume = load_nifti_image_bgr(file_path)
    tam_x, tam_y, tam_z, _ = volume.shape
    votes_volume = np.zeros((tam_x, tam_y, tam_z))
    
    slices_x = [volume[i,:,:] for i in range(tam_x)]
    slices_y = [volume[:,j,:] for j in range(tam_y)]
    slices_z = [volume[:,:,k] for k in range(tam_z)]

    model = YOLO(model=yolo_model_path, task="segment", verbose=False)
    predictions_x = model(slices_x, save=False, verbose=True, show_boxes=False)
    predictions_y = model(slices_y, save=False, verbose=True, show_boxes=False)
    predictions_z = model(slices_z, save=False, verbose=True, show_boxes=False)
    
    for index, prediction_x in enumerate(predictions_x):
        masks = prediction_x.masks
        stack = stack_masks(masks, votes_volume[index,:,:].shape)
        votes_volume[index,:,:] = votes_volume[index,:,:] + stack

    for index, prediction_y in enumerate(predictions_y):
        masks = prediction_y.masks
        stack = stack_masks(masks, votes_volume[:,index,:].shape)
        votes_volume[:,index,:] = votes_volume[:,index,:] + stack
    
    for index, prediction_z in enumerate(predictions_z):
        masks = prediction_z.masks
        stack = stack_masks(masks, votes_volume[:,:,index].shape)
        votes_volume[:,:,index] = votes_volume[:,:,index] + stack
    
    return votes_volume

if __name__ == "__main__":
    time1 = time.time()

    file_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training/MSLesSeg-Dataset/train/P1/T1/P1_T1_FLAIR.nii"
    yolo_model_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training/runs/yolov8n-seg-me-kfold-5/weights/best.pt"
    votes = consensus(file_path, yolo_model_path)

    time2 = time.time()
    
    print("For time: ", time2-time1)
    print(votes.min(), votes.max())
