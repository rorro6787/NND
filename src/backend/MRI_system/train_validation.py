from ultralytics import YOLO
import os
import pandas as pd
import cv2
import numpy as np
from scan import ImageScan
import torch
import csv

def train_YOLO(name_model: str, yaml_file_path: str, path=os.getcwd()) -> None:
    """
    Trains a YOLOv8 segmentation model using a specified dataset.

    This function loads a pre-trained YOLOv8n model for segmentation and
    trains it using the dataset specified in the provided YAML file. It
    creates a directory for saving the training results if it does not
    already exist.

    Args:
        name_model (str): The name to assign to the training experiment, which
                          will be used for saving the model.
        yaml_file_path (str): The path to the YAML file that contains dataset
                              configuration details (e.g., paths to training
                              and validation images, class names).
        path (str, optional): The directory path where results will be saved.
                              Defaults to the current working directory if not
                              provided.

    Returns:
        None: This function does not return any value. It initiates the training
              process for the YOLO model.

    Example:
        train_YOLO("yolov8n-seg-experiment", "/path/to/dataset.yaml")

    Note:
        The function uses the pre-trained YOLOv8n segmentation model from the
        file 'yolov8n-seg.pt'. Ensure that this file is available in the
        working directory or specify its path directly in the model loading
        section if it's located elsewhere.
    """

    # Load the pre-trained YOLOv8n model for segmentation
    model = YOLO("yolov8n-seg.pt", task="segmentation")

    save_directory = os.path.join(path, "runs")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Train the model with your dataset
    model.train(
        data=yaml_file_path,       # Path to the YAML file with data configuration
        epochs=128,                # Number of training epochs
        imgsz=320,                 # Image size (width and height)
        # lr0=0.01,                # Initial learning rate
        # lrf=0.001,               # Final learning rate
        # weight_decay=0.0005,     # Weight decay for regularization
        # momentum=0.937,          # Momentum for SGD optimizer
        # dampening=0.5,           # Momentum damping
        # nesterov=True,           # Use Nesterov momentum
        # accumulative=2,          # Gradient accumulation steps
        batch=-1,                  # Batch size, -1 for default
        name=name_model,           # Experiment/model name
        device=0,                  # Device ID for training (0 for first GPU)
        project=save_directory,    # Project directory for results
        save_dir=save_directory,   # Directory to save the trained model
        fraction=0.5,              # Fraction of dataset for training
        # hyp=None,                # Hyperparameter file path or None for defaults
        # local_rank=-1,           # Local GPU rank for distributed training
        # sync_bn=False,           # Use synchronized batch norm
        # workers=8,               # Number of data loading workers
        plots=True,                # Generate training plots
        # freeze=[0],              # Freeze specific layers (list of layer indices)
        save_period=1,             # Save model every 'n' epochs
        # resume=False,            # Resume training from a saved model
        # val=True,                # Validate model after each epoch
        # image_weights=False,     # Weight images in loss
        # hyp_path=None,           # Path to hyperparameter file
        # save_json=True,          # Save results as JSON
        # lr_schedule=True,        # Use learning rate scheduling
        # rect=False,              # Use rectangular image resizing
        # single_cls=False,        # Train on single class only
        # compute_map=False,       # Calculate mAP during validation
        # iou = 0.9,               # IoU threshold for mAP calculation
        # conf = 0.2,              # Confidence threshold for mAP calculation
    )

def train_kfolds_YOLO(path: str = os.getcwd()) -> None:
    """
    Trains a YOLOv8 segmentation model using k-fold cross-validation.

    This function runs the training process for a YOLOv8 segmentation model
    five times, each time with a different fold of the dataset. The model
    names and the corresponding YAML configuration files are generated based
    on the current fold index.

    Args:
        path (str): The directory path where the dataset YAML files are located.
                     Defaults to the current working directory if not provided.

    Returns:
        None: This function does not return any value. It initiates the training
              process for each fold of the dataset.

    Example:
        train_kfolds_YOLO()  # Trains the model using the current working directory
        train_kfolds_YOLO('/path/to/dataset')  # Trains using the specified path

    Note:
        The function assumes that YAML files named 'MSLesSeg_Dataset-0.yaml',
        'MSLesSeg_Dataset-1.yaml', etc., exist in the specified path, corresponding
        to each fold of the dataset.
    """

    # It would be interesting to parallelize this process when using Picasso
    for i in range(5):
        name_model = f"yolov8n-seg-me-kfold-{i+1}"
        yaml_file_path = os.path.join(
            path, "k_fold_configs", f"MSLesSeg_Dataset-{i+1}.yaml"
        )
        train_YOLO(name_model, yaml_file_path, path=path)

def obtain_model_metrics(output_path: str) -> dict:
    results_path = os.path.join(output_path, "results.csv")
    results_csv = pd.read_csv(results_path)
    print(results_csv)

def segment_image(paths: list, model_file: str = "yolov8n-seg-me.pt") -> str:
    """
    Perform segmentation on input images using a YOLO model and save the results to a specified directory.

    This function takes a list of image file paths, processes them using a YOLO segmentation model, 
    and saves the segmented images to the specified output directory.

    Parameters:
    -----------
    paths : list
        A list of strings, where each string is the path to an image file to be processed.
    output_directory : str
        The directory where the segmented images and results will be saved.
    model_file : str, optional
        The filename of the YOLO model to use for segmentation. 
        Default is 'yolov8n-seg-me.pt', which is a lightweight model suitable for segmentation tasks.
    
    Returns:
    --------
    list
        A list of paths to the segmented images saved in the output directory.

    Example:
    --------
    >>> results = segment_image(["/path/to/image1.jpg", "/path/to/image2.jpg"], "/path/to/output/")
    >>> print(results)
    ['/path/to/output/image1_segmented.png', '/path/to/output/image2_segmented.png']
    
    This will perform segmentation on image1.jpg and image2.jpg, saving the results in the specified output directory
    and returning the paths of the segmented images.
    """

    print(model_file)

    # Define the full path to the model file based on the current working directory
    model_path = os.path.join(os.getcwd(), model_file)
    
    # Load the YOLO model for the segmentation task
    model = YOLO(model=model_path, task="segment", verbose=False)

    # Initialize an empty list to hold the results of the segmentation
    results = []

    # Run the model on the current batch and save the results
    with torch.no_grad(): 
        try:
            results.extend(
                model(
                    paths, save=False, verbose=True, show_boxes=False
                )
            )

            # del batch_results
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch: {e}")

    return results

def stack_masks(mask_list: list, image_shape: tuple) -> np.ndarray:
    """
    Combine multiple binary masks into a single binary image by resizing and stacking them.
    
    Args:
        mask_list (list): A list of binary masks (PyTorch tensors) that need to be stacked.
        image_shape (tuple): The target shape of the output binary image, defined as (height, width).
    
    Returns:
        np.ndarray: A single binary image where the individual masks are combined using a logical OR operation.
    
    Notes:
        - The function first converts the input masks from PyTorch tensors to NumPy arrays.
        - Each mask is resized to the target `image_shape` using OpenCV.
        - The masks are then stacked (combined) using a pixel-wise logical OR operation.
    """

    # Convert the list of masks from PyTorch tensor to NumPy arrays
    masks = mask_list.data.cpu().numpy()
    
    # Initialize a blank binary image with the specified shape
    binary_image = np.zeros(image_shape)
    
    # Resize and combine each mask into the binary image
    for mask in masks:
        resized_mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
        binary_image = np.logical_or(binary_image, resized_mask)
    
    return binary_image.astype(np.uint8)

def calculate_metrics(metrics: dict, ground_truth_mask: np.ndarray, predicted_mask: np.ndarray) -> dict:
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

    if predicted_mask is None:
        predicted_mask = np.zeros_like(ground_truth_mask)

    total_pixels = ground_truth_mask.size
    TP = np.sum(np.logical_and(ground_truth_mask, predicted_mask))  
    FP = np.sum(np.logical_and(np.logical_not(ground_truth_mask), predicted_mask))  
    TN = np.sum(np.logical_and(np.logical_not(ground_truth_mask), np.logical_not(predicted_mask))) 
    FN = total_pixels - (TP + FP + TN)  

    metrics["TP"] += TP
    metrics["FP"] += FP
    metrics["TN"] += TN
    metrics["FN"] += FN
    return metrics
      
def compare_model_mask(scans: list, fold: str, model_file: str="yolov8n-seg-me.pt") -> dict:
    """
    Compares segmentation masks generated by a model with expected masks for a set of image scans.

    This function processes image scans, generates segmentation masks using a model, and compares
    these masks against ground truth masks, saving the results in a specified output directory.

    Parameters:
        scans (list): A list of `ImageScan` instances, each containing an image and its ground truth mask.

    Returns:
        dict: A dictionary containing pixel-level statistics (TP, FP, TN, FN) for model performance evaluation.

    The function follows these steps:
    1. Constructs paths for the test images.
    2. Creates an output directory for saving results if it doesn't exist.
    3. Segments each image in batches, retrieving segmentation results.
    4. For each scan, compares generated masks against expected masks and calculates pixel-level statistics.
    5. Saves combined images (original, generated mask, expected mask) in the output directory.

    Raises:
        OSError: If the output directory cannot be created or an image cannot be processed.
    """

    # Define the path to the test images
    image_directory = os.path.join(os.getcwd(), "MSLesSeg-Dataset-a", fold, "images")
    
    # Collect image paths for each scan object
    image_paths = [scan.get_image_path(image_directory) for scan in scans]

    # Create a directory to save output images if it does not exist
    output_directory = os.path.join(os.getcwd(), "pruebas_yolo")
    os.makedirs(output_directory, exist_ok=True)

    # Batch size for processing images to avoid memory overload
    batch_size = 128
    metrics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for i in range(0, len(image_paths), batch_size):
        # Get the current batch of image paths
        batch_paths = image_paths[i:i + batch_size]
        
        # Segment images using the model
        results = segment_image(batch_paths, model_file=model_file)
        # print(torch.cuda.memory_summary())

        for index, result in enumerate(results):
            masks = result.masks
            expected_mask = scans[index].obtain_image_mask(os.getcwd())
            mask_image = stack_masks(masks, expected_mask.shape) if masks else None
            metrics = calculate_metrics(metrics, expected_mask, mask_image)  
            # cv2.imwrite(os.path.join(output_directory, f"{scans[index].get_image_name()}_mx.png"), combined_masks)
    
    formulas = calculate_formulas(metrics)
    return (metrics, formulas)

def calculate_formulas(metrics: dict) -> dict:
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
    
    return { "Recall": recall, "Precision": precision, "Acc": acc, "Sensibility": sensibility, "IOU": iou, "F1 Score": f1_score }
    
def tests(fold: str, image: str = None, model_file: str = "yolov8n-seg-me.pt") -> None:
    image_directory = os.path.join(os.getcwd(), "MSLesSeg-Dataset-a", fold, "images")
    tests = []
    if not image:
        for file in os.listdir(image_directory):
            if file.endswith(".png"):
                scan = ImageScan(image_directory, file)
                tests.append(scan)   
    else:
        scan = ImageScan(image_directory, image)
        tests = [scan]

    metrics_over_time = []

    for i in range(0, 127, 18):
        metrics = compare_model_mask(tests, fold, model_file=os.path.join(model_file, f"epoch{i}.pt"))
        metrics_over_time.append(metrics)
    
    metrics = compare_model_mask(tests, fold, model_file=os.path.join(model_file, "best.pt"))
    metrics_over_time.append(metrics)
    write_csv(metrics_over_time, os.path.dirname(model_file))

def write_csv(metrics: dict, path: str) -> None:
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

    header = ['TP', 'FP', 'TN', 'FN', 'Recall', 'Precision', 'Acc', 'Sensibility', 'IOU', 'F1 Score']
    # Write the metrics to a CSV file
    with open(os.path.join(path, 'output.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(header)
        
        # Write the metrics for each item in the dictionary
        for item in metrics:
            row = list(item[0].values()) + list(item[1].values()) 
            writer.writerow(row)

    print("CSV escrito correctamente.")

def combine_masks(image_path: str, ground_truth_mask, predicted_mask=None) -> np.ndarray:
    image = cv2.imread(image_path)

    if predicted_mask is None:
        predicted_mask = np.zeros_like(ground_truth_mask)

    COLOR_PURPLE = [255, 0, 255]   
    COLOR_RED = [0, 0, 255]        
    COLOR_BLUE = [255, 0, 0]         

    image_with_masks_rgba = np.copy(image)

    for i in range(image_with_masks_rgba.shape[0]):
        for j in range(image_with_masks_rgba.shape[1]):
            if predicted_mask[i, j] and ground_truth_mask[i, j]:
                image_with_masks_rgba[i, j] = COLOR_PURPLE
            elif predicted_mask[i, j] and not ground_truth_mask[i, j]:
                image_with_masks_rgba[i, j] = COLOR_BLUE
            elif not predicted_mask[i, j] and ground_truth_mask[i, j]:
                image_with_masks_rgba[i, j] = COLOR_RED

    return image_with_masks_rgba

def get_purple(image: str, fold: str, model_file: str="yolov8n-seg-me.pt") -> None:
    image_directory = os.path.join(os.getcwd(), "MSLesSeg-Dataset-a", fold, "images")
    image_path = os.path.join(image_directory, image)
    if not image.endswith(".png"):
        return
    
    scan = ImageScan(image_directory, image)
    scan = scan.obtain_image_mask(os.getcwd())
    results = segment_image(image_path, model_file=model_file)[0].masks
    mask_image = stack_masks(results, scan.shape) if results else None
    purple_mask = combine_masks(image_path, scan, mask_image)

    desired_width = 600
    desired_height = 450
    resized_image = cv2.resize(purple_mask, (desired_width, desired_height))

    cv2.imshow("Purple Mask", resized_image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(os.getcwd(), f"{image}_mx.png"), purple_mask)
    
if __name__ == "__main__":
    # print(ultralytics.checks())
    # train_kfolds_YOLO()
    # i = 1
    # name_model = f"yolov8n-seg-me-kfold-{i}"
    # yaml_file_path = os.path.join(os.getcwd(), "k_fold_configs", f"MSLesSeg_Dataset-{i}.yaml")
    # train_YOLO(name_model, yaml_file_path, path=os.getcwd())
    # model_metrics = obtain_model_metrics(os.path.join(os.getcwd(), "runs", name_model))
    # compare_model_mask("P41_T1_FLAIR_axial_80")
    
    image = "P41_T1_FLAIR_axial_74.png"
    model_file = os.path.join(os.getcwd(), "runs", "yolov8n-seg-me-kfold-3", "weights")
    # tests(image=image, model_file=model_file)
    tests(fold="fold5", image=None, model_file=model_file)
    # get_purple(image, model_file=model_file)





