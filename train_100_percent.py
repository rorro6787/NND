#!/usr/bin/env python3
"""
train_100_percent.py
====================
Script to train each model using 100% of the training dataset (MSLesSeg 53 patients).
Models supported (10 configurations from the paper):
  - nnUNet3D (nnU-Net 3d_fullres, fold all)
  - nnUNet2D (nnU-Net 2d, fold all)
  - Yolo3D   (YOLOv11x-seg multi-plane 3D, trained on 100% of axial+coronal+sagittal slices)
  - Yolo3D-a (YOLOv11x-seg axial slice model, 100% data)
  - Yolo3D-c (YOLOv11x-seg coronal slice model, 100% data)
  - Yolo3D-s (YOLOv11x-seg sagittal slice model, 100% data)
  - Yolo2D   (YOLOv11x-seg 2D multi-plane consensus ensemble: trains axial, coronal, sagittal models on 100% data)
  - Yolo2D-a (YOLOv11x-seg axial single-plane model, 100% data)
  - Yolo2D-c (YOLOv11x-seg coronal single-plane model, 100% data)
  - Yolo2D-s (YOLOv11x-seg sagittal single-plane model, 100% data)

Usage:
  python train_100_percent.py --model all
  python train_100_percent.py --model nnUNet3D
  python train_100_percent.py --model Yolo3D --epochs 100 --batch 24
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# Add current directory to path if needed
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from nnd.logger import get_logger
from nnd.models.nnUNet.__init__ import Configuration as NN_Configuration, Fold as NN_Fold, Trainer as NN_Trainer
from nnd.models.yolo.__init__ import YoloModel, Trainer as Yolo_Trainer
from nnd.utils.utils_dataset import MSLESSEG_TRAIN_DATASET_URL, MSLESSEG_TEST_DATASET_URL

logger = get_logger("train_100_percent")

ALL_MODELS = [
    "nnUNet3D",
    "nnUNet2D",
    "Yolo3D",
    "Yolo3D-a",
    "Yolo3D-c",
    "Yolo3D-s",
    "Yolo2D",
    "Yolo2D-a",
    "Yolo2D-c",
    "Yolo2D-s",
]


def generate_full_yolo_yaml(dataset_path: str, trainer_type: str, output_yaml_path: str) -> str:
    """
    Generate a YAML file containing 100% of the training data (folds 1 through 5).
    """
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    
    train_paths = []
    # Include all 5 folds into training
    for fold_idx in range(1, 6):
        fold_dir = os.path.join(dataset_path, "MSLesSeg-Dataset-a", f"fold{fold_idx}")
        if not os.path.exists(fold_dir):
            fold_dir = os.path.join(dataset_path, "MSLesSeg-Dataset-YOLO", f"fold{fold_idx}")
        
        if trainer_type == Yolo_Trainer.FULL_3D.value:
            for plane in ["axial", "coronal", "sagittal"]:
                train_paths.append(os.path.join(fold_dir, plane, "images"))
        else:
            train_paths.append(os.path.join(fold_dir, trainer_type, "images"))

    data_config = {
        "train": train_paths,
        "val": train_paths,  # evaluate on full training set during training checks
        "nc": 1,
        "names": ["multiple_esclerosis"],
    }

    with open(output_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    logger.info(f"Generated 100% data YAML config at {output_yaml_path}")
    return output_yaml_path


def train_yolo_100pct(
    trainer_type: str,
    yolo_model_name: str = "yolo11x-seg.pt",
    epochs: int = 100,
    batch_size: int = 24,
    dataset_path: str = ".",
    output_dir: str = "yolo_trainings_100pct",
) -> None:
    """
    Train a YOLO model on 100% of the dataset slices for a given orientation or FULL_3D.
    """
    try:
        from ultralytics import YOLO
        from nnd.models.yolo.train_augm import _train_parameters, _augmentation_parameters
    except ImportError:
        raise ImportError("ultralytics package is required to train YOLO models. Run: pip install -e .")

    project_dir = os.path.join(output_dir, yolo_model_name.removesuffix(".pt"), trainer_type)
    os.makedirs(project_dir, exist_ok=True)
    yaml_path = os.path.join(project_dir, f"full_train_{trainer_type}.yaml")
    generate_full_yolo_yaml(dataset_path, trainer_type, yaml_path)

    train_params = _train_parameters()
    augm_params = _augmentation_parameters()
    params = {**train_params, **augm_params}

    params["epochs"] = epochs
    params["batch"] = batch_size
    if trainer_type in (Yolo_Trainer.FULL_3D.value, Yolo_Trainer.SIMPLE_CORONAL.value):
        params["imgsz"] = 256
    else:
        params["imgsz"] = 192

    logger.info(f"Starting 100% data training for YOLO ({trainer_type}, imgsz={params['imgsz']})...")
    model = YOLO(yolo_model_name, task="segmentation")
    model.train(
        data=yaml_path,
        project=project_dir,
        name="full_100pct",
        **params,
    )
    logger.info(f"Finished 100% data training for YOLO ({trainer_type}). Saved to {project_dir}/full_100pct")


def train_nnunet_100pct(
    dataset_id: str = "024",
    configuration: str = "3d_fullres",
    trainer: str = "nnUNetTrainer_100epochs",
) -> None:
    """
    Train nnU-Net on 100% of the training data using fold 'all'.
    """
    logger.info(f"Starting 100% data training for nnU-Net (Dataset {dataset_id}, {configuration}, fold all)...")
    cmd = [
        "nnUNetv2_train",
        dataset_id,
        configuration,
        "all",
        "-tr",
        trainer,
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Finished 100% data training for nnU-Net ({configuration}, fold all).")


def main():
    parser = argparse.ArgumentParser(
        description="Train segmentation models using 100% of the MSLesSeg training dataset."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=["all"] + ALL_MODELS,
        help="Model to train on 100%% data. Default is 'all'.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="024",
        help="nnUNet Dataset ID (default: 024).",
    )
    parser.add_argument(
        "--nnunet_trainer",
        type=str,
        default="nnUNetTrainer_100epochs",
        help="nnUNet trainer class (default: nnUNetTrainer_100epochs).",
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolo11x-seg.pt",
        help="YOLO model checkpoint or weights (default: yolo11x-seg.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for YOLO training (default: 100).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size for YOLO training (default: 24).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Path to root containing MSLesSeg datasets (default: current directory).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="train_100pct_results",
        help="Directory to save 100%% training models and weights.",
    )

    args = parser.parse_args()

    selected = ALL_MODELS if args.model == "all" else [args.model]
    logger.info(f"Selected models for 100% data training: {selected}")
    logger.info(f"Train Dataset Reference: {MSLESSEG_TRAIN_DATASET_URL}")
    logger.info(f"Official Test Dataset Reference: {MSLESSEG_TEST_DATASET_URL}")

    # Track distinct trainings executed to avoid duplicate runs
    executed_yolo = set()

    for model_name in selected:
        logger.info(f"\n==========================================")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"==========================================")

        if model_name == "nnUNet3D":
            train_nnunet_100pct(
                dataset_id=args.dataset_id,
                configuration="3d_fullres",
                trainer=args.nnunet_trainer,
            )

        elif model_name == "nnUNet2D":
            train_nnunet_100pct(
                dataset_id=args.dataset_id,
                configuration="2d",
                trainer=args.nnunet_trainer,
            )

        elif model_name in ("Yolo3D", "Yolo3D-a", "Yolo3D-c", "Yolo3D-s"):
            # Yolo3D multi-plane consensus and single-plane 3D use the FULL_3D trained model
            if Yolo_Trainer.FULL_3D.value not in executed_yolo:
                train_yolo_100pct(
                    trainer_type=Yolo_Trainer.FULL_3D.value,
                    yolo_model_name=args.yolo_model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    dataset_path=args.data_dir,
                    output_dir=args.output_dir,
                )
                executed_yolo.add(Yolo_Trainer.FULL_3D.value)
            else:
                logger.info(f"YOLO {Yolo_Trainer.FULL_3D.value} 100% model already trained in this session.")

        elif model_name in ("Yolo2D", "Yolo2D-a", "Yolo2D-c", "Yolo2D-s"):
            # For 2D variants, train axial, coronal, and/or sagittal on 100% data
            planes_to_train = []
            if model_name == "Yolo2D":
                planes_to_train = [Yolo_Trainer.SIMPLE_AXIAL.value, Yolo_Trainer.SIMPLE_CORONAL.value, Yolo_Trainer.SIMPLE_SAGITTAL.value]
            elif model_name == "Yolo2D-a":
                planes_to_train = [Yolo_Trainer.SIMPLE_AXIAL.value]
            elif model_name == "Yolo2D-c":
                planes_to_train = [Yolo_Trainer.SIMPLE_CORONAL.value]
            elif model_name == "Yolo2D-s":
                planes_to_train = [Yolo_Trainer.SIMPLE_SAGITTAL.value]

            for plane in planes_to_train:
                if plane not in executed_yolo:
                    train_yolo_100pct(
                        trainer_type=plane,
                        yolo_model_name=args.yolo_model,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        dataset_path=args.data_dir,
                        output_dir=args.output_dir,
                    )
                    executed_yolo.add(plane)
                else:
                    logger.info(f"YOLO 2D ({plane}) 100% model already trained in this session.")

    logger.info("\n100% data training workflow completed successfully.")


if __name__ == "__main__":
    main()
