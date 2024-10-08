from ultralytics import YOLO
import os

def train_YOLO(name_model: str, yaml_file_path: str, path = os.getcwd()) -> None:
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
    model = YOLO('yolov8n-seg.pt')  

    save_directory = os.path.join(path, 'runs')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Train the model with your dataset
    model.train(
        data=yaml_file_path,       # Specify the path to the YAML file
        epochs=24,                 # Number of epochs, adjust based on your preference
        imgsz=320,                 # Image size
        batch=-1,                  # Batch size, adjust based on your GPU
        name=name_model,           # Name of the experiment
        device=0,                  # Training device, 0 for the first GPU
        project=save_directory,    # Specify the project directory
        save_dir=save_directory    # Specify the save directory for the trained model          
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
        yaml_file_path = os.path.join(path, "k_fold_configs", f"MSLesSeg_Dataset-{i+1}.yaml")
        train_YOLO(name_model, yaml_file_path, path=path)

def tryYOLO(image_name: str, output_path: str, model="yolov8n-seg-me.pt") -> str:
    model_path = os.path.join(os.getcwd(), model)
    model = YOLO(model=model_path, task="segment", verbose=False)
    img = os.path.join(image_name, image_name)
    results = model(img, save = True, project=output_path, verbose=False, show_boxes=False)
    return results[0].save_dir

if __name__ == '__main__':
    train_kfolds_YOLO()
