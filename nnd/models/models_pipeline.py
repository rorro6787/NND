import os
import sys
from nnd.models.nnUNet.__init__ import Configuration as NN_CONFIGURATION, Fold as NN_Fold, Trainer as NN_Trainer
from nnd.models.nnUNet.nnUNet_pipeline import nnUNet
from nnd.models.yolo.yolo_pipeline import yolo_init
from nnd.models.yolo.__init__ import YoloModel, Trainer as Yolo_Trainer, Validator as Yolo_Validator

# ------------------------------------------------------------------------------
# USER CONFIGURATION (modify these as needed)
# ------------------------------------------------------------------------------

# nnUNet settings
DATASET_ID        = "024"                           # e.g. "024"
NNUNET_CONFIG     = NN_CONFIGURATION.FULL_3D       # e.g. FULL_3D or any other Configuration
NNUNET_TRAINER    = NN_Trainer.EPOCHS_100           # e.g. EPOCHS_100 (must match available Trainer enum)
NNUNET_CSV_PATH   = "nnunet_all_results.csv"        # where to aggregate all nnUNet results

# YOLO settings
YOLO_MODEL        = YoloModel.V11X_SEG              # e.g. the YOLO backbone/version
YOLO_TRAINER      = Yolo_Trainer.FULL_3D            # e.g. FULL_3D training regimen
YOLO_VALIDATOR    = Yolo_Validator.A2D              # e.g. single‐plane axial 2D validator
YOLO_CONSENSUS_T  = 2                               # consensus threshold for YOLO‐validation
# ------------------------------------------------------------------------------

def run_nnunet_all_folds():
    """
    Launch nnUNet training & evaluation for all 5 folds.
    Results (Dice/IoU) for each fold will be appended into NNUNET_CSV_PATH.
    """
    # If the CSV doesn't exist yet, create a header row first
    if not os.path.isfile(NNUNET_CSV_PATH):
        with open(NNUNET_CSV_PATH, "w") as f:
            f.write("Algorithm,Stage,Metric,ExecutionID,Value\n")

    for fold_enum in NN_Fold:
        print(f"\n--- Starting nnUNet: dataset {DATASET_ID}, config={NNUNET_CONFIG.name}, fold={fold_enum.name} ---\n")
        # Instantiate nnUNet with the chosen dataset/config/fold/trainer
        pipeline = nnUNet(
            dataset_id    = DATASET_ID,
            configuration = NNUNET_CONFIG,
            fold          = fold_enum,
            trainer       = NNUNET_TRAINER,
        )
        # Call execute_pipeline, which will download data, preprocess, train, infer, evaluate, and append to CSV
        pipeline.execute_pipeline(NNUNET_CSV_PATH)
        print(f"--- Completed nnUNet fold {fold_enum.name} ---\n")


def run_yolo_all_folds():
    """
    Launch YOLO k‐fold training & validation for fold‐IDs "000" to "004".
    """
    for i in range(5):
        fold_id = f"{i:03d}"  # "000", "001", "002", "003", "004"
        print(f"\n--- Starting YOLO: fold_id={fold_id}, model={YOLO_MODEL.name} ---\n")

        # yolo_init will:
        # 1. Ensure raw dataset is downloaded
        # 2. Ensure YOLO‐formatted dataset is present (either generates or downloads)
        # 3. Train k‐fold YOLO (creating weights under ./yolo_trainings/{fold_id}/fold_{1..5}/weights)
        # 4. Validate each fold and print confusion matrices & metrics
        yolo_init(
            fold_id             = fold_id,
            yolo_model          = YOLO_MODEL,
            trainer             = YOLO_TRAINER,
            validator           = YOLO_VALIDATOR,
            consensus_threshold = YOLO_CONSENSUS_T,
        )
        print(f"--- Completed YOLO fold {fold_id} ---\n")


if __name__ == "__main__":
    # 1) Run nnUNet for all folds
    run_nnunet_all_folds()

    # 2) Run YOLO for all fold‐IDs
    run_yolo_all_folds()

    print("\nAll trainings finished successfully.\n")
