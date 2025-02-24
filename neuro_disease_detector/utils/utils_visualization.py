from vedo import Volume, show

def show_volumes(scan: str, mask: str, prediction: str) -> None:
    """
    Show the volumes in a 3D plot

    Args:
        scan: The volume to show
        mask: The mask volume to show
        prediction: The model's prediction volume
    
    Returns:
        None

    Example:
        >>> from neuro_disease_detector.utils.utils_visualization import show_volumes
        >>>
        >>> # Define the paths to the volumes
        >>> scan = scan_path
        >>> mask = mask_path
        >>> prediction = prediction_path
        >>>
        >>> # Show the volumes
        >>> show_volumes(scan, mask, prediction)
    """
    
    scan = Volume(scan)
    mask = Volume(mask).cmap("Reds").add_scalarbar("Ground Truth")
    prediction = Volume(prediction).cmap("Greens").add_scalarbar("Model Prediction", pos=(0.1, 0.06))

    show(scan, mask, prediction, axes=1)
