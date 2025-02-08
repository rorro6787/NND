from vedo import Volume, show
import os

cwd = os.getcwd()

timepoints_patient = [3,4,4,3,2,3,   
                      2,2,3,2,2,4,2,  
                      4,1,1,1,1,4,3,1,1,
                      2,1,1,1,1,2,1,0,2,1,2,1,1,1,   
                      1,1,1,1,1,1,1,2,2,2,2,1,1,     
                      1,1,1,2]

def get_patient_by_test_id(test_id: int | str):
    """ 
    Given a test ID and a list with the number of tests per patient,
    return the patient to which the test belongs.

    Args:
        test_id (int | str): The ID of the test.

    Returns:
        str: The patient to which the test belongs.
    """

    test_id = int(test_id)
    current_id = 0

    for i, num_tests in enumerate(timepoints_patient):
        current_id += num_tests
        if test_id <= current_id:
            return f"P{i + 1}"
        
    return "ID not found"

def show_volumes(vol: str, mask: str, prediction: str):
    """
    Show the volumes in a 3D plot

    Args:
        vol: The volume to show
        mask: The mask volume to show
        prediction: The model's prediction volume
    
    Returns:
        None
    """
    
    vol = Volume(vol)
    mask = Volume(mask).cmap("Reds").add_scalarbar("Ground Truth")
    prediction = Volume(prediction).cmap("Greens").add_scalarbar("Model Prediction", pos=(0.1, 0.06))

    show(vol, mask, nnUNet_prediction, axes=1)

if __name__ == "__main__":
    nnUNet_raw = f"{cwd}/nnUNet_raw/Dataset024_MSLesSeg"
    nnUNet_prediction = f"{nnUNet_raw}/nnUNet_tests_0/BRATS_89.nii.gz"
    mask = f"{nnUNet_raw}/labelsTs/BRATS_89.nii.gz"
    flair = f"{nnUNet_raw}/imagesTs/BRATS_89_0000.nii.gz"
    show_volumes(flair, mask, nnUNet_prediction)
