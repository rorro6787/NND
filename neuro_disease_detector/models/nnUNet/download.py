import pandas as pd
import zipfile
import gdown
import os

def download_nnunet_cloud(dataset_zip: str = os.getcwd()) -> None:
    """Downloads and extracts a dataset from a cloud storage URL."""

    print(f"Downloading nnUNet dataset from Cloud...")
    url = "https://drive.google.com/uc?export=download&id=160SzQ_iG9j5M5qohYwA7iX9zzgGm1fR0"
    gdown.download(url, dataset_zip, quiet=False)
    
def prepare_nnunet_dataset(id: str, path: str = os.getcwd()):
    """ 
    Prepares the nnUNet dataset by checking if the dataset zip file exists. 
    If not, it downloads the dataset. Then, it extracts the dataset to the 
    specified folder if not already extracted.
    """

    dataset_zip = f"{path}/nnu_net.zip"
    dataset_folder = f"{path}/nnu_net"
      
    if not os.path.exists(dataset_zip):
        download_nnunet_cloud(dataset_zip)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder, exist_ok=True)

    # Extract the dataset from the ZIP file
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        print(f"Extracting dataset with id: {id} to Local")
        zip_ref.extractall(dataset_folder)

    os.rename(f"{dataset_folder}/nnUNet_preprocessed/Dataset_MSLesSeg", f"{dataset_folder}/nnUNet_preprocessed/Dataset{id}_MSLesSeg")
    os.rename(f"{dataset_folder}/nnUNet_raw/Dataset_MSLesSeg", f"{dataset_folder}/nnUNet_raw/Dataset{id}_MSLesSeg")

for i in range(3):
    prepare_nnunet_dataset(f"00{i}")