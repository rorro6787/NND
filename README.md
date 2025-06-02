**Bachelor’s Thesis – University of Málaga (Sept 2024 – Jun 2025)**
**Author:** Emilio Rodrigo Carreira Villalta ([emiliorodrigo.ecr@gmail.com](mailto:emiliorodrigo.ecr@gmail.com))

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rorro6787/neurodegenerative-disease-detector)
[![PyPI Version](https://img.shields.io/pypi/v/nnd.svg)](https://pypi.org/project/nnd/)
[![PyPI Python version](https://img.shields.io/pypi/pyversions/nnd.svg)](https://pypi.org/project/nnd/)
[![PyPI License](https://img.shields.io/pypi/l/nnd.svg)](https://pypi.org/project/nnd/)
[![CI](https://github.com/rorro6787/neurodegenerative-disease-detector/actions/workflows/test.yml/badge.svg)](https://github.com/rorro6787/neurodegenerative-disease-detector/actions/workflows/test.yml)

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

### 2. Launch experiments

```bash
cd models

# Train the different YOLO and nnUNet models
python models_pipeline.py
```

### 3. Ensemble inference

An example Jupyter notebook that visualises the different nnUNet model inferences is available at `notebooks/predict/visualize_segmentation.ipynb`.

### 4. Statistical analysis

You can observe the statistical analysis results at `notebooks/results/experimental_results.ipynb`.

## Repository structure

```text
├── .github/                  # Github Actions
├── information/              # Relevant Documents
├── neuro_disease_detector/   # Main Library
├── notebooks/                # Exploratory notebooks
├── picasso/                  # Picasso files used for Experiments
├── tests/                    # Test for the package
├── .gitignore          
├── LICENSE.md
├── pyproject.toml
└── README.md
```

## Results

See `information/ThesisTFG.pdf` for the full ablation study.

## Citing

If you use this code, please cite:

```bibtex
@article{carreira2025sclerosis,
  author  = {Emilio Rodrigo Carreira Villalta},
  title   = {Deep Learning–Based MRI Segmentation for Sclerosis Using YOLOv11 and nnUNet},
  journal = {To appear},
  year    = {2025}
}
```

## License

This project is licensed under the MIT License – see `LICENSE.md` for details.

## Acknowledgements

* University of Málaga & SCENIC research group for guidance.
* Málaga Supercomputing Center (*Picasso*) for GPU resources.
* The open‑source communities behind YOLO, nnUNet and PyTorch.
