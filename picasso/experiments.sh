#!/bin/bash

#SBATCH -J test_gpus.sh
#SBATCH --time=168:00:00
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=dgx
##SBATCH --constraint=cal

# Set output and error files
#SBATCH --error=logs/test_gpus.%J.err
#SBATCH --output=logs/test_gpus.%J.out

date
hostname

module purge
module load python

# Save current directory and move to scratch space
pushd .
cd $LOCALSCRATCH

# Create user directory if it doesn't exist
if ! test -d ${USER} ; then
    mkdir ${USER}
    mkdir ${USER}/data
fi

cd ${USER}/data

# Ensure that the dataset is extracted in the correct directory
if pwd | grep $LOCALSCRATCH >/dev/null ; then
    if ! test -d $FSCRATCH/MSLesSeg-Dataset ; then
        time tar xfz ~/data/MSLesSeg-Dataset.tar.gz
    fi
else
    echo "ERROR: we are not in $LOCALSCRATCH, we are in `pwd`"
fi

# Check the size and number of files to confirm correct dataset extraction
du -ks $LOCALSCRATCH/${USER}/data
find $LOCALSCRATCH/${USER}/data | wc

pwd
echo "Returning to the original directory after preparing the data"
popd
pwd

cd neurodegenerative-disease-detector/models
time python models_pipeline.py
