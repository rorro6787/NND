from ultralytics import YOLO
import os
import pandas as pd
import cv2
import numpy as np
from scan import ImageScan

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
        epochs=16,                 # Number of training epochs
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
        fraction=0.1,              # Fraction of dataset for training
        # hyp=None,                # Hyperparameter file path or None for defaults
        # local_rank=-1,           # Local GPU rank for distributed training
        # sync_bn=False,           # Use synchronized batch norm
        # workers=8,               # Number of data loading workers
        plots=True,                # Generate training plots
        # freeze=[0],              # Freeze specific layers (list of layer indices)
        # save_period=1,           # Save model every 'n' epochs
        # resume=False,            # Resume training from a saved model
        # val=True,                # Validate model after each epoch
        # image_weights=False,     # Weight images in loss
        # hyp_path=None,           # Path to hyperparameter file
        # save_json=True,          # Save results as JSON
        # lr_schedule=True,        # Use learning rate scheduling
        # rect=False,              # Use rectangular image resizing
        # single_cls=False,        # Train on single class only
        # compute_map=False,       # Calculate mAP during validation
        iou = 0.9,                 # IoU threshold for mAP calculation
        conf = 0.2,                # Confidence threshold for mAP calculation
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

def segment_image(image_file: str, output_directory: str, model_file: str = "yolov8n-seg-me.pt") -> str:
    """
    Perform segmentation on an input image using a YOLO model and save the result.

    Parameters:
    -----------
    image_file : str
        The name of the image file (including its path) to be processed.
    output_directory : str
        The directory where the segmented image and results will be saved.
    model_file : str, optional
        The YOLO model file to use for segmentation. Default is 'yolov8n-seg-me.pt'.
    
    Returns:
    --------
    str
        The path to the first result (segmented image).
    """

    # Define the full path to the model file based on the current working directory
    model_path = os.path.join(os.getcwd(), model_file)
    
    # Load the YOLO model for segmentation task
    model = YOLO(model=model_path, task="segment", verbose=False)
    
    # Define the full path to the input image
    image_path = os.path.join(image_file)

    # Perform segmentation on the image, save the result, and set the output project directory
    results = model(
        image_path, save=True, project=output_directory, verbose=True, show_boxes=False
    )

    # Return the path of the first result (segmented image)
    return results[0]

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
    
    return binary_image

def combine_masks(image_path: str, ground_truth_mask, predicted_mask=None):
    """
    Combines an original image with two binary masks (ground truth and predicted) 
    to visualize their overlaps with different colors.

    Parameters:
        image_path (str): Path to the original image.
        ground_truth_mask (np.ndarray): Binary mask representing the ground truth.
        predicted_mask (np.ndarray, optional): Binary mask representing the predicted segmentation.

    Returns:
        np.ndarray: Image with combined masks in RGBA format, where:
            - Overlap of both masks is colored purple with transparency.
            - Pixels only in the predicted mask are colored red with transparency.
            - Pixels only in the ground truth mask are colored blue with transparency.
    """
    image = cv2.imread(image_path)

    if predicted_mask is None:
        predicted_mask = np.zeros_like(ground_truth_mask)

    # Define colors in RGB
    COLOR_PURPLE = [255, 0, 255]   # Purple  
    COLOR_RED = [255, 0, 0]        # Red  
    COLOR_BLUE = [0, 0, 255]       # Blue  

    # Convert the original image to RGBA format
    image_with_masks_rgba = np.copy(image)

    # Apply colors with transparency based on the matches between the masks
    for i in range(image_with_masks_rgba.shape[0]):
        for j in range(image_with_masks_rgba.shape[1]):
            if predicted_mask[i, j] and ground_truth_mask[i, j]:
                # If both masks are present, pixel is purple with transparency
                image_with_masks_rgba[i, j] = COLOR_PURPLE
            elif predicted_mask[i, j] and not ground_truth_mask[i, j]:
                # If only the predicted mask is present, pixel is red with transparency
                image_with_masks_rgba[i, j] = COLOR_BLUE
            elif not predicted_mask[i, j] and ground_truth_mask[i, j]:
                # If only the ground truth mask is present, pixel is blue with transparency
                image_with_masks_rgba[i, j] = COLOR_RED

    # Return the resulting image with transparency
    return image_with_masks_rgba

def compare_model_mask(scan: ImageScan) -> None:
    """
    Compares the model's segmentation masks with the expected masks for a given image scan.

    Parameters:
        scan (ImageScan): An instance of ImageScan containing information about the image 
                          and its expected segmentation mask.

    Returns:
        None
    """
    # Define the path to the test images
    image_directory = os.path.join(os.getcwd(), "MSLesSeg-Dataset-a", "test", "images")
    image_path = scan.get_image_path(image_directory)

    # Create a directory to save output images if it does not exist
    output_directory = os.path.join(os.getcwd(), "pruebas_yolo")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Segment the image and get the results
    results = segment_image(image_path, output_directory)
    save_directory, masks = results.save_dir, results.masks

    # Obtain the expected mask and prepare it for saving
    expected_mask = scan.obtain_image_mask(os.getcwd())
    expected_mask_image = (expected_mask * 255).astype(np.uint8)
    expected_mask_image = np.clip(expected_mask_image, 0, 255)
    expected_mask_path = os.path.join(save_directory, f"{scan.get_image_name()}_m.png")
    cv2.imwrite(expected_mask_path, expected_mask_image)

    # Copy the original image to the save directory
    os.system(f"cp {image_path} {os.path.join(save_directory, f'{scan.get_image_name()}_og.png')}")

    combined_masks = None

    if not masks:
        combined_masks = combine_masks(image_path, expected_mask_image)
    else:
        # Stack the generated masks into a single image
        mask_image = stack_masks(masks, expected_mask_image.shape)
        combined_masks = combine_masks(image_path, expected_mask_image, mask_image)

    # Combine the mask image with the original image and expected mask, then save it
    cv2.imwrite(os.path.join(save_directory, f"{scan.get_image_name()}_mx.png"), combined_masks)

    return save_directory

def delete_files(directory: str) -> None:
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path) and not file.endswith("mx.png"):
            os.remove(path)

def tests():
    for _ in range(100):
        patient = np.random.randint(42, 53)
        timespan = 1
        modality = np.random.randint(0, 3)
        type_slice = np.random.randint(0, 3)
        slice_number = np.random.randint(40, 160)
        scan = ImageScan(patient, timespan, slice_number, modality, type_slice)
        dir = compare_model_mask(scan)
        delete_files(dir)

if __name__ == "__main__":
    # print(ultralytics.checks())
    # train_kfolds_YOLO()
    # i = 1
    # name_model = f"yolov8n-seg-me-kfold-{i}"
    # yaml_file_path = os.path.join(os.getcwd(), "k_fold_configs", f"MSLesSeg_Dataset-{i}.yaml")
    # train_YOLO(name_model, yaml_file_path, path=os.getcwd())
    # model_metrics = obtain_model_metrics(os.path.join(os.getcwd(), "runs", name_model))

    # compare_model_mask("P41_T1_FLAIR_axial_80")
    tests()

    # obtain_image_mask("P41_T1_FLAIR_axial_80")
    # model = YOLO("yolov8n-seg-me.pt", task="segmentation")
    # yaml = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/MRI-Neurodegenerative-Disease-Detection/src/backend/MRI_system/k_fold_configs/MSLesSeg_Dataset-1.yaml"
    # model.val(data=yaml, batch=50, save_json=True, project=os.path.join(os.getcwd(), "runs_extra"), conf=0.2, iou=0.9, plots=True)
    
    