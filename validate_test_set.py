#!/usr/bin/env python3
"""
validate_test_set.py
====================
Script to evaluate each model on the official MSLesSeg test dataset (22 held-out test patients).
Official Test Dataset Figshare URL:
https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209

Models supported (10 configurations from the paper):
  - nnUNet3D (nnU-Net 3d_fullres)
  - nnUNet2D (nnU-Net 2d)
  - Yolo3D   (YOLOv11x-seg 3D multi-plane consensus ensemble, threshold >= 2)
  - Yolo3D-a (YOLOv11x-seg 3D axial single plane)
  - Yolo3D-c (YOLOv11x-seg 3D coronal single plane)
  - Yolo3D-s (YOLOv11x-seg 3D sagittal single plane)
  - Yolo2D   (YOLOv11x-seg 2D multi-plane consensus ensemble [axial+coronal+sagittal], threshold >= 2)
  - Yolo2D-a (YOLOv11x-seg 2D axial single plane)
  - Yolo2D-c (YOLOv11x-seg 2D coronal single plane)
  - Yolo2D-s (YOLOv11x-seg 2D sagittal single plane)

Usage:
  python validate_test_set.py --model all --test_dir ./MSLesSeg-Test
  python validate_test_set.py --model Yolo3D --weights_3d ./yolo_trainings_100pct/yolo11x-seg/3d_fullres/full_100pct/weights/best.pt
  python validate_test_set.py --model nnUNet3D --test_dir ./MSLesSeg-Test
"""

import os
import sys
import glob
import argparse
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    import nibabel as nib
except ImportError:
    nib = None

from nnd.logger import get_logger
logger = get_logger("validate_test_set")

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


def stack_masks_2d(masks_list, target_shape):
    """Combine predicted 2D polygons/masks into a single binary slice."""
    import cv2
    if not masks_list:
        return np.zeros(target_shape, dtype=np.uint8)
    arrs = []
    for m in masks_list:
        nm = m.data.cpu().numpy() if hasattr(m, "data") else m
        arrs.append(nm)
    stacked = cv2.resize(arrs[0], (target_shape[1], target_shape[0]))
    for nm in arrs[1:]:
        resized = cv2.resize(nm, (target_shape[1], target_shape[0]))
        stacked = np.logical_or(stacked, resized)
    return stacked.astype(np.uint8)


def compute_binary_metrics(prediction: np.ndarray, mask: np.ndarray) -> dict:
    """Compute DSC, IoU, Precision, Recall between binary 3D prediction and ground truth."""
    tp = np.sum((mask == 1) & (prediction == 1))
    tn = np.sum((mask == 0) & (prediction == 0))
    fp = np.sum((mask == 0) & (prediction == 1))
    fn = np.sum((mask == 1) & (prediction == 0))

    recall = float(np.nan_to_num(tp / (tp + fn), nan=0.0))
    precision = float(np.nan_to_num(tp / (tp + fp), nan=0.0))
    iou = float(np.nan_to_num(tp / (tp + fn + fp), nan=0.0))
    dsc = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn), nan=0.0))

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "DSC": dsc,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
    }


def find_test_cases(test_dir: str):
    """
    Locates all test subjects in test_dir.
    Supports either hierarchical (P{id}/T{tp}/*.nii) or flat folders.
    """
    cases = []
    # Search for hierarchical structure
    patient_dirs = sorted(glob.glob(os.path.join(test_dir, "P*")))
    if patient_dirs:
        for p_dir in patient_dirs:
            p_name = os.path.basename(p_dir)
            tp_dirs = sorted(glob.glob(os.path.join(p_dir, "T*")))
            if tp_dirs:
                for t_dir in tp_dirs:
                    t_name = os.path.basename(t_dir)
                    flair = glob.glob(os.path.join(t_dir, "*FLAIR.nii*"))
                    mask = glob.glob(os.path.join(t_dir, "*MASK.nii*"))
                    t1 = glob.glob(os.path.join(t_dir, "*T1.nii*"))
                    t2 = glob.glob(os.path.join(t_dir, "*T2.nii*"))
                    if flair:
                        cases.append({
                            "id": f"{p_name}_{t_name}",
                            "flair": flair[0],
                            "t1": t1[0] if t1 else None,
                            "t2": t2[0] if t2 else None,
                            "mask": mask[0] if mask else None,
                        })
            else:
                flair = glob.glob(os.path.join(p_dir, "*FLAIR.nii*"))
                mask = glob.glob(os.path.join(p_dir, "*MASK.nii*"))
                if flair:
                    cases.append({
                        "id": p_name,
                        "flair": flair[0],
                        "t1": None,
                        "t2": None,
                        "mask": mask[0] if mask else None,
                    })

    if not cases:
        # Search for flat NIfTI files
        flairs = sorted(glob.glob(os.path.join(test_dir, "*FLAIR*.nii*")))
        for f in flairs:
            base = os.path.basename(f)
            case_id = base.split("_FLAIR")[0].split(".")[0]
            mask_candidates = glob.glob(os.path.join(test_dir, f"{case_id}*MASK*.nii*"))
            cases.append({
                "id": case_id,
                "flair": f,
                "t1": None,
                "t2": None,
                "mask": mask_candidates[0] if mask_candidates else None,
            })

    return cases


def evaluate_yolo_model(
    model_name: str,
    test_cases: list,
    weights_path: dict | str,
    output_dir: str,
) -> pd.DataFrame:
    """
    Run evaluation on the test cases for a specific YOLO model configuration.
    """
    logger.info(f"Evaluating {model_name} on {len(test_cases)} test cases...")

    try:
        from ultralytics import YOLO
        from nnd.models.yolo.validation_consensus import yolo_3d_prediction, _yolo_3d_prediction
        from nnd.models.yolo.utils.utils_nifti import load_nifti_image, load_nifti_image_bgr
    except ImportError as e:
        raise ImportError(f"Required ML dependencies missing: {e}. Please run inside your virtual environment.")

    # Load YOLO weights
    is_3d = model_name.startswith("Yolo3D")
    if is_3d:
        model_3d = YOLO(weights_path if isinstance(weights_path, str) else weights_path["3d"], task="segmentation", verbose=False)
        models_2d = {}
    else:
        model_3d = None
        models_2d = {
            plane: YOLO(pt, task="segmentation", verbose=False)
            for plane, pt in weights_path.items()
            if os.path.exists(pt)
        }

    results = []

    for case in test_cases:
        case_id = case["id"]
        flair_path = case["flair"]
        mask_path = case["mask"]

        flair_bgr = load_nifti_image_bgr(flair_path)
        tam_x, tam_y, tam_z, _ = flair_bgr.shape
        votes = np.zeros((tam_x, tam_y, tam_z), dtype=np.int32)

        # Decide consensus threshold
        if model_name in ("Yolo3D", "Yolo2D"):
            cth = 2
        else:
            cth = 1

        # Run inference
        if is_3d:
            preds_x, preds_y, preds_z = yolo_3d_prediction(flair_bgr, model_3d)

            if model_name == "Yolo3D":
                for i, p in enumerate(preds_x):
                    votes[i, :, :] += stack_masks_2d(p.masks, (tam_y, tam_z))
                for j, p in enumerate(preds_y):
                    votes[:, j, :] += stack_masks_2d(p.masks, (tam_x, tam_z))
                for k, p in enumerate(preds_z):
                    votes[:, :, k] += stack_masks_2d(p.masks, (tam_x, tam_y))
            elif model_name == "Yolo3D-a":
                for k, p in enumerate(preds_z):
                    votes[:, :, k] += stack_masks_2d(p.masks, (tam_x, tam_y))
            elif model_name == "Yolo3D-c":
                for j, p in enumerate(preds_y):
                    votes[:, j, :] += stack_masks_2d(p.masks, (tam_x, tam_z))
            elif model_name == "Yolo3D-s":
                for i, p in enumerate(preds_x):
                    votes[i, :, :] += stack_masks_2d(p.masks, (tam_y, tam_z))

        else:
            # 2D models
            if model_name in ("Yolo2D", "Yolo2D-s") and "sagittal" in models_2d:
                slices_x = [flair_bgr[i, :, :] for i in range(tam_x)]
                preds_x = _yolo_3d_prediction(models_2d["sagittal"], slices_x)
                for i, p in enumerate(preds_x):
                    votes[i, :, :] += stack_masks_2d(p.masks, (tam_y, tam_z))

            if model_name in ("Yolo2D", "Yolo2D-c") and "coronal" in models_2d:
                slices_y = [flair_bgr[:, j, :] for j in range(tam_y)]
                preds_y = _yolo_3d_prediction(models_2d["coronal"], slices_y)
                for j, p in enumerate(preds_y):
                    votes[:, j, :] += stack_masks_2d(p.masks, (tam_x, tam_z))

            if model_name in ("Yolo2D", "Yolo2D-a") and "axial" in models_2d:
                slices_z = [flair_bgr[:, :, k] for k in range(tam_z)]
                preds_z = _yolo_3d_prediction(models_2d["axial"], slices_z)
                for k, p in enumerate(preds_z):
                    votes[:, :, k] += stack_masks_2d(p.masks, (tam_x, tam_y))

        pred_binary = (votes >= cth).astype(np.uint8)

        # Save prediction volume
        out_nii_path = os.path.join(output_dir, model_name, f"{case_id}_pred.nii.gz")
        os.makedirs(os.path.dirname(out_nii_path), exist_ok=True)
        if nib is not None:
            flair_nii = nib.load(flair_path)
            pred_nii = nib.Nifti1Image(pred_binary, flair_nii.affine, flair_nii.header)
            nib.save(pred_nii, out_nii_path)

        # Calculate metrics if ground truth mask exists
        case_res = {"Model": model_name, "Case": case_id}
        if mask_path and os.path.exists(mask_path):
            gt_mask = load_nifti_image(mask_path)
            metrics = compute_binary_metrics(pred_binary, gt_mask)
            case_res.update(metrics)
        else:
            case_res.update({"DSC": np.nan, "IoU": np.nan, "Precision": np.nan, "Recall": np.nan})

        results.append(case_res)

    return pd.DataFrame(results)


def evaluate_nnunet_model(
    model_name: str,
    test_dir: str,
    output_dir: str,
    dataset_id: str = "024",
    trainer: str = "nnUNetTrainer_100epochs",
) -> pd.DataFrame:
    """
    Run evaluation for nnUNet (2D or 3D_fullres) using nnUNetv2_predict.
    """
    config = "3d_fullres" if model_name == "nnUNet3D" else "2d"
    preds_out = os.path.join(output_dir, model_name, "predictions")
    os.makedirs(preds_out, exist_ok=True)

    logger.info(f"Running nnU-Net inference ({model_name}, config={config})...")
    # Execute nnUNetv2_predict
    cmd = [
        "nnUNetv2_predict",
        "-i", test_dir,
        "-o", preds_out,
        "-d", dataset_id,
        "-c", config,
        "-tr", trainer,
        "-f", "all",
    ]
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.warning(f"nnUNetv2_predict failed or command not found in PATH: {e}")

    # Evaluate predictions against test ground truth if masks available
    test_cases = find_test_cases(test_dir)
    results = []
    for case in test_cases:
        case_id = case["id"]
        pred_candidates = glob.glob(os.path.join(preds_out, f"{case_id}*.nii*"))
        case_res = {"Model": model_name, "Case": case_id}
        if pred_candidates and case["mask"] and os.path.exists(case["mask"]):
            pred_mask = load_nifti_image(pred_candidates[0])
            gt_mask = load_nifti_image(case["mask"])
            metrics = compute_binary_metrics(pred_mask, gt_mask)
            case_res.update(metrics)
        else:
            case_res.update({"DSC": np.nan, "IoU": np.nan, "Precision": np.nan, "Recall": np.nan})
        results.append(case_res)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Validate segmentation models on the official MSLesSeg test dataset."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=["all"] + ALL_MODELS,
        help="Model to validate. Default is 'all'.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./MSLesSeg-Test",
        help="Path to the official test dataset directory containing the 22 held-out cases.",
    )
    parser.add_argument(
        "--weights_3d",
        type=str,
        default="./yolo_trainings_100pct/yolo11x-seg/3d_fullres/full_100pct/weights/best.pt",
        help="Path to YOLO 3D fullres weights (.pt).",
    )
    parser.add_argument(
        "--weights_axial",
        type=str,
        default="./yolo_trainings_100pct/yolo11x-seg/axial/full_100pct/weights/best.pt",
        help="Path to YOLO axial weights (.pt).",
    )
    parser.add_argument(
        "--weights_coronal",
        type=str,
        default="./yolo_trainings_100pct/yolo11x-seg/coronal/full_100pct/weights/best.pt",
        help="Path to YOLO coronal weights (.pt).",
    )
    parser.add_argument(
        "--weights_sagittal",
        type=str,
        default="./yolo_trainings_100pct/yolo11x-seg/sagittal/full_100pct/weights/best.pt",
        help="Path to YOLO sagittal weights (.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_validation_output",
        help="Output directory for test predictions and CSV metrics.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="test_set_evaluation_results.csv",
        help="File to save final aggregated test results across all models.",
    )

    args = parser.parse_args()

    selected = ALL_MODELS if args.model == "all" else [args.model]
    logger.info(f"Selected models for validation: {selected}")
    logger.info(f"Official Test Dataset Figshare URL: {MSLESSEG_TEST_DATASET_URL}")

    os.makedirs(args.output_dir, exist_ok=True)
    test_cases = find_test_cases(args.test_dir)
    logger.info(f"Discovered {len(test_cases)} cases in {args.test_dir}")

    all_dfs = []

    for model_name in selected:
        logger.info(f"\n==========================================")
        logger.info(f"Validating model: {model_name}")
        logger.info(f"==========================================")

        if model_name in ("nnUNet3D", "nnUNet2D"):
            df = evaluate_nnunet_model(
                model_name=model_name,
                test_dir=args.test_dir,
                output_dir=args.output_dir,
            )
            all_dfs.append(df)

        elif model_name.startswith("Yolo3D"):
            df = evaluate_yolo_model(
                model_name=model_name,
                test_cases=test_cases,
                weights_path=args.weights_3d,
                output_dir=args.output_dir,
            )
            all_dfs.append(df)

        elif model_name.startswith("Yolo2D"):
            weights_dict = {
                "axial": args.weights_axial,
                "coronal": args.weights_coronal,
                "sagittal": args.weights_sagittal,
            }
            df = evaluate_yolo_model(
                model_name=model_name,
                test_cases=test_cases,
                weights_path=weights_dict,
                output_dir=args.output_dir,
            )
            all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        csv_out_path = os.path.join(args.output_dir, args.output_csv)
        final_df.to_csv(csv_out_path, index=False)
        logger.info(f"\nSaved detailed evaluation to: {csv_out_path}")

        # Summary statistics
        if "DSC" in final_df.columns and not final_df["DSC"].isna().all():
            summary = final_df.groupby("Model")[["DSC", "IoU", "Precision", "Recall"]].agg(["mean", "std"])
            logger.info("\n=== Test Set Benchmark Summary (Mean ± Std) ===")
            print(summary.to_string())


if __name__ == "__main__":
    main()
