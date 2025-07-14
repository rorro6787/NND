from vedo import Volume, show
import os
import sys

def show_volumes(scan: str, mask: str, prediction: str = None) -> None:
    """
    Show the volumes in a 3D plot

    Args:
        scan: The volume to show
        mask: The mask volume to show
        prediction: The model's prediction volume or None
    
    Returns:
        None

    Example:
        >>> show_volumes(scan, mask, prediction)
    """
    scan_vol = Volume(scan)
    mask_vol = Volume(mask).cmap("Reds").add_scalarbar("Ground Truth")
    
    if prediction != "raw":
        prediction_vol = Volume(prediction).cmap("Greens").add_scalarbar("Model's Prediction", pos=(0.1, 0.06))
        show(scan_vol, mask_vol, prediction_vol, axes=1)
    else:
        show(scan_vol, mask_vol, axes=1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <raw|some_prediction_suffix>")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    cwd = os.getcwd()
    validation_dir = os.path.join(cwd, "validation")
    
    # Paths to scan and mask volumes (fixed here)
    scan_path = os.path.join(validation_dir, "original", "BRATS_86.nii")
    mask_path = os.path.join(validation_dir, "mask", "BRATS_86.nii")
    
    if arg1 == "raw":
        prediction_path = "raw"
    else:
        prediction_path = os.path.join(validation_dir, f"nnUNet{arg1}", "BRATS_86.nii.gz")
    
    show_volumes(scan_path, mask_path, prediction_path)
