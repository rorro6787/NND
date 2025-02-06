
num_timepoints_per_patient = [3,4,4,3,2,3,2,2,3,2,2,4,2,4,1,1,1,1,4,3,1,2,1,1,1,1,1,2,1,0,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2]

fold_to_patient = {
      "fold1": (1, 6),
      "fold2": (6, 12),
      "fold3": (12, 19),
      "fold4": (19, 28),
      "fold5": (28, 41),
      "test": (41, 54)
}

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


