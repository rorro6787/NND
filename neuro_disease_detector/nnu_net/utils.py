import os
from neuro_disease_detector.nnu_net.nnUNet_pipeline import _split_assign

num_timepoints_per_patient = [3,4,4,3,2,3,2,2,3,2,2,4,2,4,1,1,1,1,4,3,1,2,1,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2]

fold_to_patient = {
      "fold1": (1, 7),
      "fold2": (7, 14),
      "fold3": (14, 23),
      "fold4": (23, 42),
      "fold5": (42, 54),
}


def patients(dataset_path: str = f"{os.getcwd()}/MSLesSeg-Dataset/train"):
    timepoints = []
    for pd in range(1, 54):
        if pd == 30:
            timepoints.append(0)
            continue
        pd_path = f"{dataset_path}/P{pd}"
        td_count = 0
        for td in range(1, 5):
            td_path = f"{pd_path}/T{td}"
            if not os.path.exists(td_path):
                break
            td_count += 1
        timepoints.append(td_count)

    print(timepoints)



def get_patient_by_test_id(test_id: int|str):
    """
    Given a test ID and a list with the number of tests per patient,
    return the patient to which the test belongs.

    Args:
        test_id (int | str): The ID of the test.
        tests_per_patient (list): A list where each element indicates the number of tests for each patient.

    Returns:
        str: The patient to which the test belongs.
    """

    # Ensure test_id is an integer (in case it's provided as a string)
    test_id = int(test_id)
    
    # Initialize a variable to track the cumulative number of tests encountered
    current_id = 0

    # Iterate through each patient and the number of tests they have
    for i, num_tests in enumerate(num_timepoints_per_patient):
        # Add the number of tests for the current patient to the cumulative count
        current_id += num_tests
        
        # If the test_id is within the range of tests for this patient, return their ID
        if test_id <= current_id:
            return f"P{i + 1}"

    # If no matching patient is found, return "ID not found"
    return "ID not found"


# print(get_patient_by_test_id(92))
"""
fold1 = ("BRATS_1", "BRATS_19")
fold2 = ("BRATS_20", "BRATS_36")
fold3 = ("BRATS_37", "BRATS_54")
fold4 = ("BRATS_55", "BRATS_75")
fold5 = ("BRATS_76", "BRATS_92")
"""

print(get_patient_by_test_id("30"))
print(_split_assign(0))