import nibabel as nib
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def show_slices():
    # Load the .nii file
    nii_file = 'path'  # Change this to your file name
    img = nib.load(nii_file)

    # Get the image data as a 3D numpy array
    data = img.get_fdata()

    # View the dimensions of the 3D image
    print("3D image dimensions:", data.shape)

    # Extract 2D projections along different axes
    # Axial projection (horizontal, top to bottom)
    axial_slice = data[:, :, data.shape[2] // 2]  # Take a section in the middle
    print("Axial projection dimensions:", axial_slice.shape)

    # Sagittal projection (side view)
    sagittal_slice = data[data.shape[0] // 2, :, :]  # Take a section in the middle

    # Coronal projection (frontal)
    coronal_slice = data[:, data.shape[1] // 2, :]  # Take a section in the middle

    # Create the figure and display the three projections
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the axial projection
    axes[0].imshow(axial_slice.T, cmap="gray", origin="lower")
    axes[0].set_title('Axial (Top view)')

    # Display the sagittal projection
    axes[1].imshow(sagittal_slice.T, cmap="gray", origin="lower")
    axes[1].set_title('Sagittal (Side view)')

    # Display the coronal projection
    axes[2].imshow(coronal_slice.T, cmap="gray", origin="lower")
    axes[2].set_title('Coronal (Frontal view)')

    plt.show()

def try_YOLOv8(image_name: str) -> str:
    # Load the YOLOv5 model
    model = YOLO(model="yolov8n.pt", task="detect", verbose=False)

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
