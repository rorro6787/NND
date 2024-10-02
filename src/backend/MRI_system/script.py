import nibabel as nib
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def show_slices(nii_file: str):
    # Load the .nii file
    img = nib.load(nii_file)

    # Obtain image data as a 3D numpy array
    data = img.get_fdata()

    # Print the dimensions of the 3D image data
    print("Dimensiones de la imagen 3D:", data.shape)

    # Axial projection (top view)
    axial_slice = data[:, :, data.shape[2] // 2]  # Take a slice in the middle
    print("Dimensiones de la proyección axial:", axial_slice.shape)

    # Sagittal projection (lateral view)
    sagittal_slice = data[data.shape[0] // 2, :, :]  # Take a slice in the middle
    print("Dimensiones de la proyección sagital:", sagittal_slice.shape)

    # Coronal projection (frontal view)
    coronal_slice = data[:, data.shape[1] // 2, :]  # Take a slice in the middle
    print("Dimensiones de la proyección coronal:", coronal_slice.shape)

    return axial_slice, sagittal_slice, coronal_slice

def try_YOLOv8(image_name: str, model="yolov8n.pt") -> str:
    # Load the YOLOv5 model
    model = YOLO(model=model, task="detect", verbose=False)

    # Load an image
    img = os.path.join(os.getcwd(), '..', 'images', image_name)

    # Perform inference
    results = model(img, save = True, project=os.path.join(os.getcwd(), '..', 'images'), verbose=False)
    
    # Return the path to the saved image
    return results[0].save_dir

if __name__ == "__main__":
    image_name = "download.jpeg"
    original_image = os.path.join(os.getcwd(), '..', 'images', image_name)
    YOLO_image_path = os.path.join(try_YOLOv8(image_name), image_name)
