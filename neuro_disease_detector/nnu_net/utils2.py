
import os

fold_to_patient = {
      "fold1": (1, 7),
      "fold2": (7, 14),
      "fold3": (14, 23),
      "fold4": (23, 37),
      "fold5": (37, 50),
      "test": (50, 54)
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






# print(get_patient_by_test_id(92))
"""
fold1 = ("BRATS_1", "BRATS_19")
fold2 = ("BRATS_20", "BRATS_36")
fold3 = ("BRATS_37", "BRATS_53")
fold4 = ("BRATS_54", "BRATS_70")
fold5 = ("BRATS_71", "BRATS_87")
"""

fold_to_patient = {
      "fold1": (1, 7),
      "fold2": (7, 14),
      "fold3": (14, 23),
      "fold4": (23, 37),
      "fold5": (37, 50),
      "test": (50, 54)
}

# print(get_patient_by_test_id("87"))
# print(_split_assign(1))

# patients()