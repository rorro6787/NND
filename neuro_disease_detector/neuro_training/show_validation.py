from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from neuro_disease_detector.neuro_training.test_metrics import update_confusion_matrix, calculate_metrics

def combine_masks(image_path: str, ground_truth_mask, predicted_mask=None) -> np.ndarray:
    image = cv2.imread(image_path)

    if predicted_mask is None:
        predicted_mask = np.zeros_like(ground_truth_mask)

    COLOR_PURPLE = [255, 0, 255]   
    COLOR_RED = [0, 0, 255]        
    COLOR_BLUE = [255, 0, 0]       
    wrong = 0
    good = 0  

    image_with_masks_rgba = np.copy(image)

    for i in range(image_with_masks_rgba.shape[0]):
        for j in range(image_with_masks_rgba.shape[1]):
            if predicted_mask[i, j] and ground_truth_mask[i, j]:
                image_with_masks_rgba[i, j] = COLOR_PURPLE
                good += 1
            elif predicted_mask[i, j] and not ground_truth_mask[i, j]:
                image_with_masks_rgba[i, j] = COLOR_BLUE
                wrong += 1
            elif not predicted_mask[i, j] and ground_truth_mask[i, j]:
                image_with_masks_rgba[i, j] = COLOR_RED
                wrong += 1

    return image_with_masks_rgba, wrong, good

def get_purple(image: str, fold: str, model_file: str="yolov8n-seg-me.pt") -> None:
    image_directory = os.path.join(os.getcwd(), "MSLesSeg-Dataset-a", fold, "images")
    image_path = os.path.join(image_directory, image)
    if not image.endswith(".png"):
        return
    
    scan = ImageScan(image_directory, image)
    scan = scan.obtain_image_mask(os.getcwd())
    results = segment_image(image_path, model_file=model_file)[0].masks
    mask_image = stack_masks(results, scan.shape) if results else None
    purple_mask, wrong, good = combine_masks(image_path, scan, mask_image)

    desired_width = 600
    desired_height = 450
    resized_image = cv2.resize(purple_mask, (desired_width, desired_height))

    #cv2.imshow("Purple Mask", resized_image)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()
    return resized_image, wrong, good
    cv2.imwrite(os.path.join(os.getcwd(), "pruebas_yolo", f"{image}_mx.png"), resized_image)


def tests(fold: str, image: str = None, model_file: str = "yolov8n-seg-me.pt") -> None:
    image_directory = os.path.join(os.getcwd(), "MSLesSeg-Dataset-a", fold, "images")
    tests = []
    i = 0
    if not image:
        for file in os.listdir(image_directory):
            if file.endswith(".png"):
                
                resized_image, wrong, good = get_purple(file, "fold5", model_file=model_file)
                if good == 0:
                    ratio = wrong
                else:
                    ratio = float(wrong / good)
                if ratio == 0:
                    i += 1
                    continue
                if len(tests) == 250:
                    (ratioi, file1, resized_imagei) = tests[0]
                    if ratioi < ratio:
                        tests[0] = (ratio, file, resized_image)
                        tests = sorted(tests, key=lambda x: x[0])
                        print("RATIO ANTIGUO: " + str(ratioi))
                        print("RATIO NUEVO: " + str(tests[0][0]))
                        print("RATIO INTRODUCIDO: " + str(ratio))
                else: 
                    tests.append((ratio, file, resized_image))
                    tests = sorted(tests, key=lambda x: x[0])

                i+=1
                print(len(tests))
                print(i)
                #scan = ImageScan(image_directory, file)
                #tests.append(scan)  
            if i == 10000:
                break 
    else:
        scan = ImageScan(image_directory, image)
        tests = [scan]
    
    for (ratio, file, image) in tests:
        print(ratio)
        cv2.imwrite(os.path.join(os.getcwd(), "pruebas_yolo", f"{file}_mx.png"), image)
    """
    metrics_over_time = []

    for i in range(0, 61, 3):
        metrics = compare_model_mask(tests, fold, model_file=os.path.join(model_file, f"epoch{i}.pt"))
        metrics_over_time.append(metrics)
    
    metrics = compare_model_mask(tests, fold, model_file=os.path.join(model_file, "best.pt"))
    metrics_over_time.append(metrics)
    print(metrics_over_time)
    csv_path =  os.path.join(os.path.dirname(model_file), "output.csv")
    write_csv(metrics_over_time, csv_path)
    make_graphs(csv_path, test=False)
    """



if __name__ == "__main__":
    yolo_model = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/neuro_training/runs/yolov8n-seg-me-kfold-5/weights/best.pt"
    dataset_path = "/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/neuro_disease_detector/data_processing"
    fold = "fold1"
    validation(dataset_path, fold, yolo_model)


    # print(ultralytics.checks())
    # train_kfolds_YOLO()
    """
    make_graphs("/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/src/backend/MRI_system/runs/yolov8n-seg-me-kfold-7/val.csv", False)
    
    i = 1
    name_model = f"yolov8n-seg-me-kfold-{7}"
    yaml_file_path = os.path.join(os.getcwd(), "k_fold_configs", f"MSLesSeg_Dataset-{i}.yaml")
    # train_YOLO(name_model, yaml_file_path, path=os.path.join(os.getcwd()))
    # model_metrics = obtain_model_metrics(os.path.join(os.getcwd(), "runs", name_model))
    # compare_model_mask("P41_T1_FLAIR_axial_80")
    
    # image = "P41_T1_FLAIR_axial_74.png"
    model_file = os.path.join(os.getcwd(), "runs", "yolov8n-seg-me-kfold-7", "weights")
    # tests(fold="fold5", image=image, model_file=model_file)
    tests(fold="fold1", image=None, model_file=model_file)
    # get_purple(image, fold="fold5", model_file=model_file)

    # csv_path = os.path.join(os.getcwd(), "runs", "yolov8n-seg-me-kfold-3", "output_val.csv")
    # make_graphs(csv_path, test=False)
    """

    #model = YOLO("/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/src/backend/MRI_system/runs/yolov8n-seg-me-kfold-7/weights/best.pt", task="segmentation")
    #model.val(data="/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/src/backend/MRI_system/k_fold_configs/MSLesSeg_Dataset-1.yaml", split="test", save_dir=os.getcwd(), project=os.getcwd())
    # tests(fold="fold5", image=None, model_file="/home/rodrigocarreira/MRI-Neurodegenerative-Disease-Detection/src/backend/MRI_system/runs/yolov8n-seg-me-kfold-7/weights/best.pt")
