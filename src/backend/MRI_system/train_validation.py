from ultralytics import YOLO
import os

def trainYOLO():
    # Load the pre-trained YOLOv8n model for segmentation
    model = YOLO('yolov8n-seg.pt')  

    save_directory = os.path.join(os.getcwd(), 'runs')
    data_path = os.path.join(os.getcwd(), 'MSLesSeg_Dataset-a.yaml')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Train the model with your dataset
    model.train(
        data=data_path,            # Specify the path to the YAML file
        epochs=24,                 # Number of epochs, adjust based on your preference
        imgsz=320,                 # Image size
        batch=16,                  # Batch size, adjust based on your GPU
        name='yolov8n_me',         # Name of the experiment
        device=0,                  # Training device, 0 for the first GPU
        project=save_directory,    # Specify the project directory
        save_dir=save_directory    # Specify the save directory for the trained model          
    )

def tryYOLO(image_name: str, output_path: str, model="yolov8n-seg-me.pt") -> str:
    # Load the YOLOv5 model
    model_path = os.path.join(os.getcwd(), model)

    model = YOLO(model=model_path, task="segment", verbose=False)
    # Load an image
    img = os.path.join(image_name, image_name)

    # Perform inference
    results = model(img, save = True, project=output_path, verbose=False, show_boxes=False)
    # Return the path to the saved image
    return results[0].save_dir

if __name__ == '__main__':
    tryYOLO('prueba.png')
