#!/bin/bash

# Step 1: Clone the repository
git clone https://github.com/rorro6787/neurodegenerative-disease-detector.git

# Step 2: Switch to the repository directory
cd neurodegenerative-disease-detector || exit

# Step 3: Create and activate virtual environment
module load python
python3 -m venv venv
# /opt/homebrew/opt/python@3.10/bin/python3.10 -m venv venv
source venv/bin/activate

# Step 4: Install requirements
pip install -e .

# Step 5: Dataset Download
# 5.1 Training Dataset (MSLesSeg 53 patients, 147 volumes)
# Google Drive link:
TRAIN_URL="https://drive.google.com/uc?export=download&id=1TM4ciSeiyl-ri4_Jn4-aMOTDSSSHM6XB"
echo "To download training dataset:"
echo "gdown ${TRAIN_URL} -O MSLesSeg-Dataset.zip && unzip MSLesSeg-Dataset.zip"

# 5.2 Official Test Dataset (MSLesSeg 22 held-out test patients)
# Figshare repository:
TEST_URL="https://springernature.figshare.com/articles/dataset/MSLesSeg_baseline_and_benchmarking_of_a_new_Multiple_Sclerosis_Lesion_Segmentation_dataset/27919209"
echo "Official Test Dataset URL (Figshare):"
echo "${TEST_URL}"
