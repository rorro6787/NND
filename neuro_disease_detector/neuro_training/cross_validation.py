from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from neuro_disease_detector.neuro_training.test_metrics import update_confusion_matrix, calculate_metrics
from neuro_disease_detector.neuro_training.test_metrics import write_csv, create_metrics_graphs
from neuro_disease_detector.neuro_training.__init__ import yolo_model

def process_batch(paths: list, yolo_model_path: str) -> list:
    """
    Perform segmentation on input images using a YOLO model and save the results to a specified directory.

    This function takes a list of image file paths, processes them using a YOLO segmentation model, 
    and saves the segmented images to the specified output directory.

    Parameters:
    -----------
    paths : list
        A list of strings, where each string is the path to an image file to be processed.
    yolo_model_path : str
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
    
    # Load the YOLO model for the segmentation task
    model = YOLO(model=yolo_model_path, task="segment", verbose=False)

    # Initialize an empty list to hold the results of the segmentation
    results = []

    # Run the model on the current batch and save the results
    with torch.no_grad():
        try:
            batch_results = model(paths, save=False, verbose=True, show_boxes=False)
            results.extend(batch_results)
            del batch_results  # Ensure intermediate variables are deleted
            torch.cuda.empty_cache()  # Free GPU cache

        except Exception as e:
            print(f"Error processing batch: {e}")

    return results

def stack_masks(masks: list, image_shape: tuple) -> np.ndarray:
    """
    Stacks a list of binary masks into a single binary image, resizing each mask to the specified image shape.

    Parameters:
    -----------
    masks : list
        A list of binary masks. Each mask is expected to be a PyTorch tensor, but it will be converted
        to a NumPy array for further processing.
    
    image_shape : tuple
        The desired shape of the output binary image, in the form (height, width). All masks will be resized
        to match this shape.
    
    Returns:
    --------
    np.ndarray
        A binary image of the specified shape where each mask has been combined using a logical OR operation.
        The output is a NumPy array of type np.uint8.
    
    Notes:
    ------
    - If the `masks` list is empty, a blank binary image (all zeros) will be returned.
    - Each mask will be resized to the shape specified by `image_shape`, and then the masks will be combined
      using a logical OR operation, meaning that if a pixel is part of any mask, it will be set to 1 in the
      output image.
    """

    # If the masks list is empty, return a blank binary image of the desired shape
    if not masks:
        return np.zeros(image_shape)

    # Convert the list of PyTorch tensors to NumPy arrays
    masks = masks.data.cpu().numpy()
    stack = cv2.resize(masks[0], (image_shape[1], image_shape[0]))
    
    # Iterate over each mask in the list
    for mask in masks[1:]:
        # Resize the mask to match the target image shape
        resized_mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
        
        # Combine the resized mask into the binary image using a logical OR operation
        stack = np.logical_or(stack, resized_mask)
    
    # Return the final binary image as an unsigned 8-bit integer array
    return stack.astype(np.uint8)

def test_batch(batch: dict, confusion_matrix: dict, yolo_model_path: str):
    """
    Test a batch of images by generating predictions using a YOLO model and updating the confusion matrix
    based on the real and predicted masks.

    Parameters:
    -----------
    batch : dict
        A dictionary where the keys are image identifiers (e.g., filenames or indices) and the values are the 
        corresponding ground truth masks for those images. Each mask is expected to be in a suitable format 
        for comparison with the predicted masks.
    
    confusion_matrix : dict
        A dictionary representing the confusion matrix, which is updated during the Test process.
        It should contain counts for true positives, false positives, true negatives, and false negatives.

    yolo_model_path : str
        The file path to the trained YOLO model, which is used to generate predictions for the batch images.

    Returns:
    --------
    dict
        The updated confusion matrix after processing the batch. It contains the counts of true positives, 
        false positives, true negatives, and false negatives for the current batch of images.
    
    Notes:
    ------
    - The function uses the `process_batch` function to generate predictions (which are assumed to be in the form 
      of masks) for each image in the batch.
    - Each predicted mask is resized to match the shape of the ground truth mask using the `stack_masks` function.
    - The confusion matrix is updated by comparing each predicted mask with the corresponding ground truth mask 
      using the `update_confusion_matrix` function.
    """
    
    # Extract image identifiers from the batch
    images = list(batch.keys())

    # Generate predictions for the images using the YOLO model
    predictions = process_batch(images, yolo_model_path)
    
    # Iterate over each prediction and its corresponding ground truth mask
    for index, prediction in enumerate(predictions):
        # Get the masks from the prediction
        masks = prediction.masks

        # Retrieve the ground truth mask for the current image
        real_mask = batch[images[index]]

        # Stack and resize the predicted masks to match the ground truth shape
        predicted_mask = stack_masks(masks, real_mask.shape)

        # Update the confusion matrix based on the comparison of real and predicted masks
        confusion_matrix = update_confusion_matrix(confusion_matrix, real_mask, predicted_mask)

    return confusion_matrix

def test_neuro_system(dataset_path: str, fold: str, yolo_model_path: str) -> dict:
    """
    Test the YOLO model on a dataset, iterating over images and their corresponding ground truth masks in
    batches, and updating the confusion matrix. The function then calculates and prints test metrics based
    on the confusion matrix.

    Parameters:
    -----------
    dataset_path : str
        The file path to the dataset. It is used to locate the image and mask directories for the given fold.
    
    fold : str
        The specific fold (subset) of the dataset to test on. It corresponds to a directory within the dataset.
    
    yolo_model_path : str
        The file path to the trained YOLO model, which is used to generate predictions for the images.

    Returns:
    --------
    None
        The function prints the calculated test metrics based on the confusion matrix.

    Notes:
    ------
    - The function processes the images in batches (with a batch size of 128).
    - For each image, the corresponding mask is loaded, and the image-mask pair is added to the test batch.
    - Once the batch reaches the specified size, it is processed by the `test_batch` function, which updates
      the confusion matrix.
    - After processing all the images in the fold, the confusion matrix is used to calculate performance metrics 
      (e.g., accuracy, precision, recall) using the `calculate_metrics` function.
    - The function prints the calculated metrics for evaluation.
    """

    # Construct the paths to the images and masks directories for the specified fold
    fold_path = os.path.join(dataset_path, f'MSLesSeg-Dataset-a/{fold}/images')
    masks_path = os.path.join(dataset_path, f'MSLesSeg-Dataset-a/masks')

    # Initialize an empty dictionary for the test batch and a batch size of 128
    batch = {}
    batch_size = 128

    # Initialize the confusion matrix with zero counts for TP, FP, TN, FN
    confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    
    # Iterate over each image in the fold's images directory
    for image in os.listdir(fold_path):
        # Skip non-PNG files
        if not image.endswith(".png"):
            continue
        
        # Split the image filename to extract parts for constructing the mask filename
        chunks = image.split("_")
        mask_path = Path(masks_path) / f"{chunks[0]}_{chunks[1]}_{chunks[3]}_{chunks[4]}"

        # Read the corresponding mask image in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Add the image and its corresponding mask to the test batch
        batch[Path(fold_path) / image] = mask

        # Once the batch reaches the specified size, test the batch and reset the batch
        if len(batch) == batch_size:
            confusion_matrix = test_batch(batch, confusion_matrix, yolo_model_path)
            batch = {}

    # Process any remaining images in the last batch
    if len(batch) > 0:
        confusion_matrix = test_batch(batch, confusion_matrix, yolo_model_path)

    # Calculate and print the evaluation metrics based on the confusion matrix
    metrics = calculate_metrics(confusion_matrix)
    return confusion_matrix | metrics 

def test_neuro_system_k_folds(training_results_path: str, dataset_path: str) -> None: 
    """
    Test a YOLO-based neuro system across K-folds and generate performance metrics over time.

    This function iterates over a predefined set of folds, evaluates the performance 
    of different YOLO model checkpoints within each fold, and produces CSV files and 
    graphs summarizing the metrics for each fold.

    Args:
        training_results_path (str): 
            Path to the directory where the results of training (organized by folds) are stored. 
            Each fold should contain a `weights` subdirectory with YOLO model checkpoint files.
        
        dataset_path (str): 
            Path to the dataset being used for evaluation. This is required to test the neuro 
            system with the YOLO models.

    Workflow:
        1. Define the folds to process (fold1, fold2, ..., fold5).
        2. For each fold:
           a. Access the `weights` directory containing YOLO checkpoint files.
           b. Iterate over all checkpoint files (`*.pt`) excluding the `best.pt` file.
           c. Test the neuro system using the `test_neuro_system` function with the dataset, 
              current fold, and model checkpoint.
           d. Collect the metrics over time for the fold.
        3. Save the collected metrics for each fold to a CSV file using the `write_csv` function.
        4. Generate and save visualizations of the metrics using the `create_metrics_graphs` function.

    Note:
        - The `test_neuro_system` function is assumed to handle the evaluation of a single 
          model on the dataset for a given fold and return relevant metrics.
        - The `write_csv` and `create_metrics_graphs` functions handle the creation of CSV 
          files and graphs, respectively.

    Returns:
        None
    """

    # Define the folds to process
    folds = ["kfold-1", "kfold-2", "kfold-3", "kfold-4", "kfold-5"]

    # Iterate over each fold in the list
    for index, fold in enumerate(folds):
        # Define the path to the current fold within the training results directory
        fold_path = Path(training_results_path) / f"{yolo_model}-{fold}"

        # Initialize an empty list to store metrics over time for the current fold
        metrics_over_time = []

        # Iterate over the YOLO model checkpoint files in the fold's weights directory
        yolo_models_path = Path(fold_path) / "weights"

        # Iterate over all YOLO model checkpoint files in the weights directory
        for model in os.listdir(yolo_models_path):
            # Skip non-checkpoint files and the best.pt file
            if(not model.endswith(".pt")) or model.endswith("best.pt"):
                continue

            # Construct the full path to the YOLO model checkpoint file
            yolo_model_path = Path(yolo_models_path) / model    

            # Test the neuro system using the current fold and YOLO model checkpoint
            metrics = test_neuro_system(dataset_path, f"fold{index+1}", yolo_model_path)

            # Append the metrics to the list for the current fold
            metrics_over_time.append(metrics)

        # Save the collected metrics for the current fold to a CSV file
        csv_path = write_csv(metrics_over_time, training_results_path)

        # Generate and save visualizations of the metrics for the current fold
        create_metrics_graphs(csv_path, training_results_path)
       
if __name__ == "__main__":
    training_results_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training/runs"
    dataset_path = "/home/rorro6787/Escritorio/Universidad/4Carrera/TFG/neurodegenerative-disease-detector/neuro_disease_detector/neuro_training"
    fold = "fold1"
    test_neuro_system_k_folds(training_results_path, dataset_path)
    