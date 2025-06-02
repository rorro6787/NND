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
