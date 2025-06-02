import os
import argparse
import numpy as np
import torch
import cv2
import nibabel as nib

from ultralytics import YOLO

# Import your existing enums / utilities from neuro_disease_detector
from nnd.models.yolo.__init__ import Validator, CM, Metrics
from nnd.models.yolo.utils.utils_nifti import load_nifti_image, load_nifti_image_bgr
from nnd.logger import get_logger

logger = get_logger(__name__)


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
    Returns a list of predictions (one per slice), where each prediction has `.masks`.
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


def stack_masks(masks_list, target_shape):
    """
    Combine a list of binary masks into a single 2D mask of shape target_shape.
    Each element in masks_list may be a PyTorch tensor or a NumPy array.
    """
    if not masks_list:
        return np.zeros(target_shape, dtype=np.uint8)
    arrs = []
    for m in masks_list:
        nm = m.data.cpu().numpy() if hasattr(m, "data") else m
        arrs.append(nm.astype(np.uint8))
    # Start with the first mask, resized to target_shape
    stacked = cv2.resize(arrs[0], (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    for nm in arrs[1:]:
        resized = cv2.resize(nm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        stacked = np.logical_or(stacked, resized)
    return stacked.astype(np.uint8)


class YoloPredictor:
    def __init__(
        self,
        model_paths,           # either a single str (3D models) or a dict {plane → str} for 2D
        validator: Validator,
        consensus_threshold: int = 2,
    ):
        """
        - model_paths:
            • if validator ∈ {Cs3D, A3D, S3D, C3D}, this is a single filepath (string) to a 3D .pt
            • if validator ∈ {A2D, S2D, C2D, Cs2D}, this is a dict mapping {"axial": "...", "coronal": "...", "sagittal": "..."}
        - validator: one of the eight Validator enum values
        - consensus_threshold: used only for Cs3D / Cs2D; ignored (force =1) for single-plane modes
        """
        self.validator = validator

        # Single‐plane modes always use threshold = 1
        if validator in {Validator.A3D, Validator.S3D, Validator.C3D,
                         Validator.A2D, Validator.S2D, Validator.C2D}:
            self.cth_eff = 1
        else:
            self.cth_eff = consensus_threshold

        if isinstance(model_paths, str):
            # 3D case: load one YOLO model for all three axes
            self.model_3d = YOLO(model_paths, task="segmentation", verbose=False)
            self.models_2d = {}
        else:
            # 2D case: load each plane separately
            self.model_3d = None
            self.models_2d = {}
            for plane, pt_path in model_paths.items():
                self.models_2d[plane] = YOLO(pt_path, task="segmentation", verbose=False)

    def predict_volume(self, input_volume_path: str, output_volume_path: str):
        """
        Load a single 3D input (NIfTI). Predict a binary mask volume via
        plane‐by‐plane YOLO inference + voting + thresholding. Save as NIfTI to output path.
        """
        # 1) Load input. Use load_nifti_image_bgr → shape (X, Y, Z, 3)
        img_bgr = load_nifti_image_bgr(input_volume_path)
        # We'll also grab affine & header for saving:
        ref_nifti = nib.load(input_volume_path)
        affine = ref_nifti.affine
        header = ref_nifti.header

        # Assume img_bgr shape is (X, Y, Z, 3)
        tam_x, tam_y, tam_z, _ = img_bgr.shape

        # 2) Run per‐plane predictions
        if self.validator in {Validator.Cs3D, Validator.A3D, Validator.S3D, Validator.C3D}:
            # 3D case: one model for all planes
            preds_x, preds_y, preds_z = yolo_3d_prediction(img_bgr, self.model_3d)

        else:
            # 2D case: build preds_x, preds_y, preds_z lists (possibly empty)
            preds_x, preds_y, preds_z = [], [], []

            # Sagittal (X slices)
            if "sagittal" in self.models_2d:
                slices_x = [img_bgr[i, :, :] for i in range(tam_x)]
                preds_x = _yolo_3d_prediction(self.models_2d["sagittal"], slices_x)

            # Coronal (Y slices)
            if "coronal" in self.models_2d:
                slices_y = [img_bgr[:, j, :] for j in range(tam_y)]
                preds_y = _yolo_3d_prediction(self.models_2d["coronal"], slices_y)

            # Axial (Z slices)
            if "axial" in self.models_2d:
                slices_z = [img_bgr[:, :, k] for k in range(tam_z)]
                preds_z = _yolo_3d_prediction(self.models_2d["axial"], slices_z)

        # 3) Accumulate votes into a (X, Y, Z) array
        votes = np.zeros((tam_x, tam_y, tam_z), dtype=np.int32)

        # For Cs3D / Cs2D: use all three planes
        if self.validator in {Validator.Cs3D, Validator.Cs2D}:
            # Sagittal votes: each preds_x[i] → stack into votes[i, :, :]
            for i, pred in enumerate(preds_x):
                mask2d = stack_masks(pred.masks, (tam_y, tam_z))
                votes[i, :, :] += mask2d

            # Coronal votes: each preds_y[j] → stack into votes[:, j, :]
            for j, pred in enumerate(preds_y):
                mask2d = stack_masks(pred.masks, (tam_x, tam_z))
                votes[:, j, :] += mask2d

            # Axial votes: each preds_z[k] → stack into votes[:, :, k]
            for k, pred in enumerate(preds_z):
                mask2d = stack_masks(pred.masks, (tam_x, tam_y))
                votes[:, :, k] += mask2d

        elif self.validator in {Validator.A3D, Validator.A2D}:
            # Only axial (Z) matters
            for k, pred in enumerate(preds_z):
                mask2d = stack_masks(pred.masks, (tam_x, tam_y))
                votes[:, :, k] += mask2d

        elif self.validator in {Validator.S3D, Validator.S2D}:
            # Only sagittal (X) matters
            for i, pred in enumerate(preds_x):
                mask2d = stack_masks(pred.masks, (tam_y, tam_z))
                votes[i, :, :] += mask2d

        elif self.validator in {Validator.C3D, Validator.C2D}:
            # Only coronal (Y) matters
            for j, pred in enumerate(preds_y):
                mask2d = stack_masks(pred.masks, (tam_x, tam_z))
                votes[:, j, :] += mask2d

        else:
            raise ValueError(f"Unsupported validator mode: {self.validator}")

        # 4) Apply consensus threshold
        binary_seg = (votes >= self.cth_eff).astype(np.uint8)

        # 5) Save as NIfTI
        seg_nifti = nib.Nifti1Image(binary_seg, affine=affine, header=header)
        nib.save(seg_nifti, output_volume_path)
        logger.info(f"Saved segmentation to {output_volume_path}")


def parse_model_paths(weights_dir: str, validator: Validator):
    """
    Given a root weights directory and a validator enum, return either:
      - a single string (path to best.pt) for 3D modes, or
      - a dict {"axial": "...", "coronal": "...", "sagittal": "..."} for 2D modes.
    Expects the following structure under weights_dir:
      • 3D: weights_dir/best.pt
      • 2D single-plane: weights_dir/axial/best.pt (or coronal/... or sagittal/...) 
      • 2D three-plane (Cs2D): weights_dir/axial/best.pt, weights_dir/coronal/best.pt, weights_dir/sagittal/best.pt
    """
    if validator in {Validator.Cs3D, Validator.A3D, Validator.S3D, Validator.C3D}:
        model_path = os.path.join(weights_dir, "best.pt")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Expected 3D model at {model_path}, but not found.")
        return model_path

    else:
        # 2D modes
        mp = {}
        if validator == Validator.A2D:
            p = os.path.join(weights_dir, "axial", "best.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Expected axial 2D model at {p}, but not found.")
            mp["axial"] = p

        elif validator == Validator.S2D:
            p = os.path.join(weights_dir, "sagittal", "best.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Expected sagittal 2D model at {p}, but not found.")
            mp["sagittal"] = p

        elif validator == Validator.C2D:
            p = os.path.join(weights_dir, "coronal", "best.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Expected coronal 2D model at {p}, but not found.")
            mp["coronal"] = p

        elif validator == Validator.Cs2D:
            # need all three planes
            for plane in ("axial", "coronal", "sagittal"):
                p = os.path.join(weights_dir, plane, "best.pt")
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Expected {plane} 2D model at {p}, but not found.")
                mp[plane] = p

        else:
            raise ValueError(f"Unrecognized 2D validator: {validator}")

        return mp


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO-based segmentation on a single 3D volume (NIfTI) and output a binary mask."
    )
    parser.add_argument(
        "--validator",
        type=str,
        required=True,
        choices=[v.name for v in Validator],
        help="Which validation mode to use: Cs3D, A3D, S3D, C3D, Cs2D, A2D, S2D, or C2D."
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        help="Path to the folder containing either:\n"
             "  • best.pt (for 3D modes), or\n"
             "  • subfolders axial/, coronal/, sagittal/ (for 2D modes), each with best.pt"
    )
    parser.add_argument(
        "--input-volume",
        type=str,
        required=True,
        help="Path to a single input NIfTI volume (e.g. P123_FLAIR.nii). Should be readable by load_nifti_image_bgr."
    )
    parser.add_argument(
        "--output-volume",
        type=str,
        required=True,
        help="Where to write the output binary segmentation (NIfTI)."
    )
    parser.add_argument(
        "--consensus-th",
        type=int,
        default=2,
        help="Consensus threshold (only used for Cs3D or Cs2D)."
    )

    args = parser.parse_args()

    # Convert validator string → Validator enum
    try:
        validator_enum = Validator[args.validator]
    except KeyError:
        raise ValueError(f"Invalid validator: {args.validator}")

    # Build model paths (either a single str or dict)
    model_paths = parse_model_paths(args.weights_dir, validator_enum)

    # Instantiate predictor and run
    predictor = YoloPredictor(
        model_paths=model_paths,
        validator=validator_enum,
        consensus_threshold=args.consensus_th
    )
    predictor.predict_volume(args.input_volume, args.output_volume)


if __name__ == "__main__":
    main()
