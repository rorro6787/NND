# Deep Learning-Based MRI Segmentation for Sclerosis

> **Bachelor’s Thesis – University of Málaga (Sept 2024 – Jun 2025)**
> **Author:** Emilio Rodrigo Carreira Villalta ([emiliorodrigo.ecr@gmail.com](mailto:emiliorodrigo.ecr@gmail.com))

Segmenting sclerosis lesions on 3‑D MRI volumes is a critical yet time‑consuming step in clinical workflows.
This repository contains the open‑source code, experiments and analysis accompanying my bachelor’s thesis, which investigates the synergy between **YOLOv11** object detection and **nnUNet** semantic segmentation, enhanced by a novel 3‑D consensus‑ensemble strategy.

## Key features

* **Dual‑model pipeline** – combines fast YOLOv11 localisation with nnUNet high‑resolution segmentation.
* **Rotation‑and‑cut consensus** – aggregates predictions across multiple axial rotations and random 3‑D crops to mitigate viewpoint bias.
* **Statistical benchmarking** – leverages the [SAES](https://github.com/jMetal/SAES) library to run non‑parametric tests and automatically generate LaTeX reports.
* **HPC‑ready** – training scripts tested on Málaga’s *Picasso* Supercomputer (SLURM job templates provided).
* **Reproducible experiments** – `configs/` YAML files capture every run; hashes are logged with Weights & Biases.

## Getting started

### 1. Clone & install

```bash
git clone https://github.com/rorro6787/neurodegenerative-disease-detector.git
cd neurodegenerative-disease-detector
python3 -m venv venv
source venv/bin/activate
```

### 2. Prepare data

1. Download the **\[dataset placeholder]** (coming soon – see `docs/dataset.md`) and convert all volumes to NIfTI.
2. Run

   ```bash
   python tools/prepare_dataset.py --root /path/to/dataset --patch-size 128
   ```

### 3. Training

```bash
# Train YOLOv11 detector
python train_yolo.py --config configs/yolo/yolo11.yaml

# Train nnUNet 3‑D segmentation (fold 0)
python train_nnunet.py --config configs/nnunet/nnunet3d.yaml --fold 0
```

### 4. Ensemble inference

```bash
python infer_ensemble.py \
   --config configs/ensemble/default.yaml \
   --input  /path/to/volumes \
   --output outputs/segmentation
```

### 5. Statistical analysis

```bash
# Generate LaTeX performance report (Friedman + Holm)
python tools/saes_report.py --csv results/metrics.csv
```

## Repository structure

```
├── configs/          # YAML experiment configs
├── data/             # ✘ git‑ignored MRI volumes & labels
├── docs/             # extra docs, figures, paper draft
├── models/           # saved checkpoints
├── notebooks/        # exploratory notebooks
├── scripts/          # SLURM & helper bash scripts
├── src/              # source code package
└── tests/            # unit & regression tests
```

## Results

| Model                       |    Dice ↑ |     IoU ↑ |     HD95 ↓ |
| --------------------------- | --------: | --------: | ---------: |
| YOLOv11 + nnUNet (ensemble) | **0.865** | **0.779** | **4.2 mm** |
| nnUNet baseline             |     0.843 |     0.748 |     5.1 mm |

See `reports/final_report.pdf` for the full ablation study.

## Citing

If you use this code, please cite:

```
@article{carreira2025sclerosis,
  author  = {Emilio Rodrigo Carreira Villalta},
  title   = {Deep Learning–Based MRI Segmentation for Sclerosis Using YOLOv11 and nnUNet},
  journal = {To appear},
  year    = {2025}
}
```

## License

This project is licensed under the MIT License – see `LICENSE` for details.

## Acknowledgements

* University of Málaga & SCENIC research group for guidance.
* Malaga Supercomputing Center (*Picasso*) for GPU resources.
* The open‑source communities behind YOLO, nnUNet and PyTorch.

---

Feel free to open an issue or pull request with improvements!
