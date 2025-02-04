

fold_to_patient = {
      "fold1": (1, 6),
      "fold2": (6, 12),
      "fold3": (12, 19),
      "fold4": (19, 28),
      "fold5": (28, 41),
      "test": (41, 54)
}

def split_assign(pd: int):
    if pd >= 1 and pd < 6:
        return "fold1"
    if pd >= 6 and pd < 12:
        return "fold2"
    if pd >= 12 and pd < 19:
        return "fold3"
    if pd >= 19 and pd < 28:
        return "fold4"
    if pd >= 28 and pd < 41:
        return "fold5"
    return "test"