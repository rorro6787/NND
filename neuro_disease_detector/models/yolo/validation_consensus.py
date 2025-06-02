import os
import re
import numpy as np
import torch
import cv2

from ultralytics import YOLO

from neuro_disease_detector.models.yolo.__init__ import CM, Metrics, Validator
from neuro_disease_detector.utils.utils_dataset import get_timepoints_patient, get_patients_split
from neuro_disease_detector.models.yolo.utils.utils_nifti import load_nifti_image, load_nifti_image_bgr
from neuro_disease_detector.logger import get_logger

logger = get_logger(__name__)

class YoloFoldValidator:
    def __init__(
        self,
        folds_directory: str,
        data_folder: str,
        validator: Validator,
        consensus_threshold: int = 2,
        k_folds: int = 5,
    ) -> None:
        """
        - folds_directory: path to the directory containing folders fold1, fold2, ..., foldk
        - data_folder: base folder under which 'MSLesSeg-Dataset/train/P{patient}' lives
        - validator: which Validator enum to use (Cs3D, A3D, S3D, C3D, Cs2D, A2D, S2D, C2D)
        - consensus_threshold: only used for multi‐plane consensus (e.g. Cs3D or Cs2D)
        - k_folds: number of folds (default=5)
        """
        self.folds_dir = folds_directory
        self.data_folder = data_folder
        self.validator = validator
        self.cth = consensus_threshold
        self.k = k_folds

        # Will be filled by validate_all_folds()
        self.cm_fold_epoch = {}
        self.metrics_fold_epoch = {}

    def validate_all_folds(self) -> None:
        """Validates all k‐folds in sequence, storing confusion matrices and metrics per fold."""
        for i in range(self.k):
            fold_name = f"fold{i+1}"
            cm_epoch, metrics_epoch = self.validate_fold(fold_name)
            self.cm_fold_epoch[fold_name] = cm_epoch
            self.metrics_fold_epoch[fold_name] = metrics_epoch

    def validate_fold(self, fold: str) -> tuple:
        """
        For the given fold (e.g. "fold1"), find and load the appropriate model(s) according to self.validator,
        then run YoloValidator on that model (or models). Returns a tuple (cm_epoch, metrics_epoch),
        where each is a dict with key "best" → its cm/metrics.
        """
        fold_path = os.path.join(self.folds_dir, fold)
        weights_root = os.path.join(fold_path, "weights")

        # Determine the “split” for this fold (this was in your original code):
        # reverse‐indexing so that fold_split = fold{k‐index+1}
        fold_index = int(fold[-1])  # e.g. if fold="fold3", fold_index=3
        fold_split = f"fold{self.k - fold_index + 1}"

        cm_epoch = {}
        metrics_epoch = {}

        # 1) If we are in a 3D validator (Cs3D, A3D, S3D, C3D), we expect exactly ONE 3D model file: weights/best.pt
        if self.validator in {
            Validator.Cs3D,
            Validator.A3D,
            Validator.S3D,
            Validator.C3D,
        }:
            model_path = os.path.join(weights_root, "best.pt")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Expected 3D model at {model_path}, but it was not found.")
            # Instantiate a single YoloValidator, run it, and store “best” → result
            yolo_validator = YoloValidator(
                model_paths=model_path,
                data_folder=self.data_folder,
                consensus_threshold=self.cth,
                fold_split=fold_split,
                validator=self.validator,
            )
            yolo_validator.process_all_patients()
            cm_epoch["best"] = yolo_validator.cm
            metrics_epoch["best"] = yolo_validator.metrics

        # 2) If we're in a 2D validator that uses three separate plane‐models (Cs2D),
        #    or a single‐plane 2D validator (A2D, S2D, C2D), we look under subfolders axial/, coronal/, sagittal/.
        elif self.validator in {
            Validator.Cs2D,
            Validator.A2D,
            Validator.S2D,
            Validator.C2D,
        }:
            # Build a dict plane→path for the needed 2D models.
            model_paths_2d = {}

            # “A2D” or “S2D” or “C2D” each only needs its single plane:
            if self.validator == Validator.A2D:
                path_axial = os.path.join(weights_root, "axial", "best.pt")
                if not os.path.isfile(path_axial):
                    raise FileNotFoundError(f"Expected axial 2D model at {path_axial}, but none found.")
                model_paths_2d["axial"] = path_axial

            elif self.validator == Validator.S2D:
                path_sagittal = os.path.join(weights_root, "sagittal", "best.pt")
                if not os.path.isfile(path_sagittal):
                    raise FileNotFoundError(f"Expected sagittal 2D model at {path_sagittal}, but none found.")
                model_paths_2d["sagittal"] = path_sagittal

            elif self.validator == Validator.C2D:
                path_coronal = os.path.join(weights_root, "coronal", "best.pt")
                if not os.path.isfile(path_coronal):
                    raise FileNotFoundError(f"Expected coronal 2D model at {path_coronal}, but none found.")
                model_paths_2d["coronal"] = path_coronal

            # “Cs2D” needs all three:
            elif self.validator == Validator.Cs2D:
                for plane in ("axial", "coronal", "sagittal"):
                    path_2d = os.path.join(weights_root, plane, "best.pt")
                    if not os.path.isfile(path_2d):
                        raise FileNotFoundError(f"Expected 2D model for plane '{plane}' at {path_2d}, but none found.")
                    model_paths_2d[plane] = path_2d

            # Now instantiate YoloValidator with that small dict:
            yolo_validator = YoloValidator(
                model_paths=model_paths_2d,
                data_folder=self.data_folder,
                consensus_threshold=self.cth,
                fold_split=fold_split,
                validator=self.validator,
            )
            yolo_validator.process_all_patients()
            cm_epoch["best"] = yolo_validator.cm
            metrics_epoch["best"] = yolo_validator.metrics

        else:
            raise ValueError(f"Unrecognized Validator: {self.validator}")

        return cm_epoch, metrics_epoch


class YoloValidator:
    def __init__(
        self,
        model_paths,  # Either a single str (3D) or a dict of plane→str (2D)
        data_folder: str,
        consensus_threshold: int,
        fold_split: str,
        validator: Validator,
    ) -> None:
        """
        - model_paths: 
            • if validator∈{Cs3D, A3D, S3D, C3D}, this is a single string to a 3D .pt
            • if validator∈{A2D, S2D, C2D, Cs2D}, this is a dict mapping {“axial”/“coronal”/“sagittal”}→model_file.pt
        - data_folder: base folder where “MSLesSeg-Dataset/train/P{patient}/T{tp}…” lives
        - consensus_threshold: int (but will be overridden to 1 for any single‐plane validator)
        - fold_split: e.g. “fold5” if validating on patients of that split
        - validator: which Validator enum we’re using right now
        """
        self.validator = validator

        # Determine effective threshold: single‐plane always 1, otherwise use provided
        if validator in {
            Validator.A3D,
            Validator.S3D,
            Validator.C3D,
            Validator.A2D,
            Validator.S2D,
            Validator.C2D,
        }:
            self.cth_eff = 1
        else:
            # Cs3D or Cs2D
            self.cth_eff = consensus_threshold

        # For 3D validators, model_paths is a single string
        if isinstance(model_paths, str):
            # Load one YOLO model for all three axes:
            self.models_3d = YOLO(model_paths, task="segmentation", verbose=False)
            self.models_2d = {}  # not used
        else:
            # model_paths is a dict str→str, e.g. {"axial": "...", "coronal": "...", "sagittal": "..."}
            self.models_3d = None
            self.models_2d = {}
            for plane, pt_path in model_paths.items():
                # Load each 2D model separately
                self.models_2d[plane] = YOLO(pt_path, task="segmentation", verbose=False)

        self.data_path = os.path.join(data_folder, "MSLesSeg-Dataset", "train")
        self.test_patients = get_patients_split(fold_split)

        # Initialize confusion matrix counters
        self.cm = {CM.TP: 0, CM.FP: 0, CM.TN: 0, CM.FN: 0}
        self.metrics = None

    def process_all_patients(self) -> None:
        """Loop over every patient in self.test_patients and process it."""
        p0, p1 = self.test_patients
        for pd in range(p0, p1):
            self._process_patient(pd)

        # Once all patients are done, compute overall metrics
        self.compute_metrics()

    def _process_patient(self, pd: int) -> None:
        """
        For patient “pd”, load each timepoint T1..T{num_tp}, load their masks and scans,
        then for each scan do plane‐specific or consensus voting per self.validator.
        """
        num_tp = get_timepoints_patient(pd)
        pd_path = os.path.join(self.data_path, f"P{pd}")

        for tp in range(1, num_tp + 1):
            tp_path = os.path.join(pd_path, f"T{tp}")
            mask_path = os.path.join(tp_path, f"P{pd}_T{tp}_MASK.nii")
            flair_path = os.path.join(tp_path, f"P{pd}_T{tp}_FLAIR.nii")
            t1_path = os.path.join(tp_path, f"P{pd}_T{tp}_T1.nii")
            t2_path = os.path.join(tp_path, f"P{pd}_T{tp}_T2.nii")

            mask = load_nifti_image(mask_path)            # ground‐truth, shape (X,Y,Z)
            flair = load_nifti_image_bgr(flair_path)      # shape (X, Y, Z, 3)
            t1 = load_nifti_image_bgr(t1_path)
            t2 = load_nifti_image_bgr(t2_path)
            scans = [flair, t1, t2]

            for scan in scans:
                # 1) get raw predictions per axis, depending on 2D vs. 3D
                if self.validator in {
                    Validator.Cs3D,
                    Validator.A3D,
                    Validator.S3D,
                    Validator.C3D,
                }:
                    # Run the single 3D model on all three axes:
                    preds_x, preds_y, preds_z = yolo_3d_prediction(scan, self.models_3d)
                else:
                    # 2D validators:
                    # We'll produce preds_x only if we have a “sagittal” model, etc.
                    tam_x, tam_y, tam_z, _ = scan.shape

                    # Prepare empty placeholders if a given plane is not in models_2d:
                    preds_x = []
                    preds_y = []
                    preds_z = []

                    # Sagittal model → run on slices_x if present
                    if "sagittal" in self.models_2d:
                        slices_x = [scan[i, :, :] for i in range(tam_x)]
                        preds_x = _yolo_3d_prediction(self.models_2d["sagittal"], slices_x)

                    # Coronal model → run on slices_y if present
                    if "coronal" in self.models_2d:
                        slices_y = [scan[:, j, :] for j in range(tam_y)]
                        preds_y = _yolo_3d_prediction(self.models_2d["coronal"], slices_y)

                    # Axial model → run on slices_z if present
                    if "axial" in self.models_2d:
                        slices_z = [scan[:, :, k] for k in range(tam_z)]
                        preds_z = _yolo_3d_prediction(self.models_2d["axial"], slices_z)

                # 2) accumulate votes into a 3D array of shape (X, Y, Z)
                scan_shape_xyz = scan.shape[:3]  # (tam_x, tam_y, tam_z)
                votes = self._accumulate_votes(scan_shape_xyz, (preds_x, preds_y, preds_z))

                # 3) apply consensus threshold (self.cth_eff)
                votes_consensus = (votes >= self.cth_eff).astype(np.uint8)

                # 4) update confusion matrix against ground‐truth mask
                self._update_cm(votes_consensus, mask)

    def _accumulate_votes(
        self,
        scan_shape: tuple,
        predictions_xyz: tuple,
    ) -> np.ndarray:
        """
        Given scan_shape = (X, Y, Z) and predictions_xyz = (preds_x, preds_y, preds_z),
        accumulate a 3D “votes” array.  Which axes to include depends on self.validator.
        """
        tam_x, tam_y, tam_z = scan_shape
        preds_x, preds_y, preds_z = predictions_xyz

        # Always start with zeros:
        votes = np.zeros((tam_x, tam_y, tam_z), dtype=np.int32)

        # Helper to stack 2D masks into a plane of the right shape:
        def stack_masks(masks_list, target_shape):
            """
            Combine a list of binary masks into a single 2D mask of shape target_shape.
            (Essentially identical to your original stack_masks, but stand‐alone here.)
            """
            if not masks_list:
                return np.zeros(target_shape, dtype=np.uint8)
            # masks_list elements have a “.data.cpu().numpy()” if they are Torch tensors; otherwise assume already numpy
            arrs = []
            for m in masks_list:
                nm = m.data.cpu().numpy() if hasattr(m, "data") else m
                arrs.append(nm)
            # First mask resized to target_shape:
            stacked = cv2.resize(arrs[0], (target_shape[1], target_shape[0]))
            for nm in arrs[1:]:
                resized = cv2.resize(nm, (target_shape[1], target_shape[0]))
                stacked = np.logical_or(stacked, resized)
            return stacked.astype(np.uint8)

        # Decide which planes to include:
        #  – For Cs3D or Cs2D: combine all three planes
        #  – For A3D or A2D: only preds_z (axial)
        #  – For S3D or S2D: only preds_x (sagittal)
        #  – For C3D or C2D: only preds_y (coronal)

        if self.validator in {Validator.Cs3D, Validator.Cs2D}:
            # 3‐plane consensus: add sagittal, coronal, axial
            # Sagittal: loop over preds_x, stack into votes[i,:,:]
            for i, pred in enumerate(preds_x):
                mask2d = stack_masks(pred.masks, (tam_y, tam_z))
                votes[i, :, :] += mask2d
            # Coronal: loop over preds_y → votes[:,j,:]
            for j, pred in enumerate(preds_y):
                mask2d = stack_masks(pred.masks, (tam_x, tam_z))
                votes[:, j, :] += mask2d
            # Axial: loop over preds_z → votes[:,:,k]
            for k, pred in enumerate(preds_z):
                mask2d = stack_masks(pred.masks, (tam_x, tam_y))
                votes[:, :, k] += mask2d

        elif self.validator in {Validator.A3D, Validator.A2D}:
            # Only axial plane, i.e. preds_z
            for k, pred in enumerate(preds_z):
                mask2d = stack_masks(pred.masks, (tam_x, tam_y))
                votes[:, :, k] += mask2d

        elif self.validator in {Validator.S3D, Validator.S2D}:
            # Only sagittal plane, i.e. preds_x
            for i, pred in enumerate(preds_x):
                mask2d = stack_masks(pred.masks, (tam_y, tam_z))
                votes[i, :, :] += mask2d

        elif self.validator in {Validator.C3D, Validator.C2D}:
            # Only coronal plane, i.e. preds_y
            for j, pred in enumerate(preds_y):
                mask2d = stack_masks(pred.masks, (tam_x, tam_z))
                votes[:, j, :] += mask2d

        else:
            raise ValueError(f"Unexpected validator in _accumulate_votes: {self.validator}")

        return votes

    def _update_cm(self, prediction: np.ndarray, mask: np.ndarray) -> None:
        """
        Given a 3D binary prediction and a 3D ground truth mask,
        increment self.cm counters (TP, TN, FP, FN).
        """
        self.cm[CM.TP] += np.sum((mask == 1) & (prediction == 1))
        self.cm[CM.TN] += np.sum((mask == 0) & (prediction == 0))
        self.cm[CM.FN] += np.sum((mask == 1) & (prediction == 0))
        self.cm[CM.FP] += np.sum((mask == 0) & (prediction == 1))

    def compute_metrics(self) -> None:
        """
        After all updates to self.cm, compute aggregated metrics:
        Recall, Precision, Accuracy, Sensibility, IoU, DSC, and store in self.metrics.
        """
        TP = self.cm[CM.TP]
        FP = self.cm[CM.FP]
        TN = self.cm[CM.TN]
        FN = self.cm[CM.FN]

        # Avoid division by zero by using np.nan_to_num(..., nan=0)
        recall = np.nan_to_num(TP / (TP + FN), nan=0.0)
        precision = np.nan_to_num(TP / (TP + FP), nan=0.0)
        acc = np.nan_to_num((TP + TN) / (TP + FP + TN + FN), nan=0.0)
        sensibility = np.nan_to_num(TP / (TP + FN + FP), nan=0.0)
        iou = np.nan_to_num(TP / (TP + FN + FP), nan=0.0)
        dsc = np.nan_to_num(2 * TP / (2 * TP + FP + FN), nan=0.0)

        self.metrics = {
            Metrics.RECALL: recall,
            Metrics.PRECISION: precision,
            Metrics.ACCUARICY: acc,
            Metrics.SENSIBILITY: sensibility,
            Metrics.IOU: iou,
            Metrics.DSC: dsc,
        }


def yolo_3d_prediction(volume: np.ndarray, yolo_model: YOLO) -> tuple:
    """
    Given a 4D volume (X, Y, Z, channels), run yolo_model on all three swept planes:
      - Sagittal: slices_x = [volume[i, :, :] for i in range(X)]
      - Coronal:  slices_y = [volume[:, j, :] for j in range(Y)]
      - Axial:   slices_z = [volume[:, :, k] for k in range(Z)]
    Returns a tuple (preds_x, preds_y, preds_z), each being a list of YOLO predictions.
    """
    tam_x, tam_y, tam_z, _ = volume.shape
    # Build lists of 2D slices
    slices_x = [volume[i, :, :] for i in range(tam_x)]
    slices_y = [volume[:, j, :] for j in range(tam_y)]
    slices_z = [volume[:, :, k] for k in range(tam_z)]

    preds_x = _yolo_3d_prediction(yolo_model, slices_x)
    preds_y = _yolo_3d_prediction(yolo_model, slices_y)
    preds_z = _yolo_3d_prediction(yolo_model, slices_z)
    return preds_x, preds_y, preds_z


def _yolo_3d_prediction(yolo_model: YOLO, slices: list, batch_size: int = 1) -> list:
    """
    Given a list of 2D slices (grayscale or RGB), run yolo_model(batch) in inference mode.
    Returns a list of predictions (one per slice).
    """
    predictions = []
    with torch.no_grad():
        for i in range(0, len(slices), batch_size):
            batch = slices[i : i + batch_size]
            preds_cuda = yolo_model(batch, save=False, verbose=False, show_boxes=False)
            predictions.extend(preds_cuda)
            del preds_cuda
            torch.cuda.empty_cache()
    return predictions


if __name__ == "__main__":
    # Example usage:

    validator_type = Validator.Cs3D  # <-- e.g. pick one of: Cs3D, A3D, S3D, C3D, Cs2D, A2D, S2D, C2D
    folds_dir = "neuro_disease_detector/models/yolo/yolo_trainings/000/yolo11x-seg/3d_fullres"
    data_folder = "neuro_disease_detector/models/yolo"
    consensus_th = 2

    validator = YoloFoldValidator(
        folds_directory=folds_dir,
        data_folder=data_folder,
        validator=validator_type,
        consensus_threshold=consensus_th,
        k_folds=5,
    )
    validator.validate_all_folds()

    print("Confusion matrices per fold:")
    print(validator.cm_fold_epoch)
    print("Metrics per fold:")
    print(validator.metrics_fold_epoch)
