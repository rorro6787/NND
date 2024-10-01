import os
import requests
import gdown
import zipfile

def download_dataset() -> str:
    """ 
    Downloads and extracts the MSLesSeg-Dataset from Google Drive if it is not already present
    in the current working directory.

    The function checks if the dataset folder already exists. If it does not, it downloads a
    ZIP file containing the dataset from a specified Google Drive URL, extracts the contents,
    and then deletes the ZIP file to save space. If the dataset is already downloaded, 
    it simply informs the user.

    Steps:
    1. Get the current working directory.
    2. Check if the dataset directory exists.
    3. If the directory does not exist:
        a. Specify the Google Drive URL to download the dataset.
        b. Set the name for the ZIP file to be saved locally.
        c. Download the ZIP file from the URL using `gdown`.
        d. Extract the contents of the ZIP file into the current directory.
        e. Remove the ZIP file after extraction to clean up.
    4. If the directory exists, inform the user that the dataset is already downloaded.

    Returns:
        str: The path to the dataset directory.
    """
    
    cd = os.getcwd()  # Get the current working directory

    # Name of the ZIP file to save locally
    zip_file = 'MSLesSeg-Dataset.zip'
    dataset_file = 'MSLesSeg-Dataset'

    if not os.path.exists(f"{cd}/MSLesSeg-Dataset"):  # Check if dataset directory exists
        # Url to download the zip file from Google Drive
        url = "https://drive.google.com/uc?export=download&id=1y55uyeo79M4Cw6eg_G9-06qU6jhJphsM"

        # Download the zip file from the URL
        print("Downloading dataset...")
        response = requests.get(url)  # Send GET request to download file
        gdown.download(url, zip_file, quiet=False)  # Download using gdown
        print(f"File downloaded as {zip_file}")

        # Extract the contents of the ZIP file
        with zipfile.ZipFile(f"{cd}/{zip_file}", 'r') as zip_ref:
            zip_ref.extractall(cd)  # Extract to current directory
        
        print("Dataset downloaded and extracted.")
        # Delete the ZIP file after extraction
        os.remove(f"{cd}/{zip_file}")
        print("Dataset downloading process finished.")

    else:
        print("Dataset already downloaded in the system...")  # Inform user if dataset exists

    # Return path to dataset directory
    return f"{cd}/{dataset_file}"  

def process_dataset(path: str):
    """
    Processes a dataset of medical images stored in a hierarchical directory structure.

    This function navigates through the training dataset located in 'MSLesSeg-Dataset/train',
    identifying patients and their respective timepoints. It searches for files with a 
    '.gz' extension, decompresses them if necessary, and collects the original compressed 
    files for deletion if their decompressed counterparts already exist.

    The directory structure expected is as follows:
    - MSLesSeg-Dataset/
        - train/
            - P1/
                - T1/
                    - file1.nii.gz
                    - file2.nii.gz
                - T2/
                    - file3.nii.gz
            - P2/
                - T1/
                    - file4.nii.gz
                    - file5.nii.gz

    The function does not take any parameters and does not return any value. Instead, it modifies
    the file system by deleting files.

    Steps:
    1. Define the path to the training dataset directory.
    2. List all directories within the training dataset, which correspond to different patients.
    3. For each patient directory:
       - List all timepoint directories.
       - For each timepoint directory:
           - List all files within the timepoint directory.
           - For each file:
               - Check if it has a '.gz' extension.
               - If a decompressed version (same name without the '.gz' suffix) exists, add the original 
                 compressed file to a set of files to delete.
               - If the decompressed file does not exist, decompress the '.gz' file using the 
                 `gunzip` command.
    4. Print out the files that will be deleted and then proceed to remove them from the file system.

    Raises:
        OSError: If there are issues with file or directory access during the execution.
    """
    
    # Step 1: Define the path to the training dataset directory
    train_cd = os.path.join(path, 'train')

    # Step 2: List all directories within the training dataset for different patients
    patients = [d for d in os.listdir(train_cd) if os.path.isdir(os.path.join(train_cd, d))]

    # Initialize a set to keep track of files to delete
    files_to_delete = set()
    
    # Step 3: Loop through each patient directory
    for patient in patients:
        print(f"Subfile: {patient}")
        patient_cd = os.path.join(train_cd, patient)  # Construct the patient's directory path
        
        # List all timepoint directories for the current patient
        timepoints = [d for d in os.listdir(patient_cd) if os.path.isdir(os.path.join(patient_cd, d))]
        
        # Loop through each timepoint directory
        for timepoint in timepoints:
            print(f"\tSubfile: {timepoint}")
            timepoint_cd = os.path.join(patient_cd, timepoint)  # Construct the timepoint's directory path
            
            # List all files within the timepoint directory
            files = os.listdir(timepoint_cd)
            
            # Loop through each file in the timepoint directory
            for file in files:
                print(f"\t\tFile: {file}")
                
                # Check if the file has a '.gz' extension
                if file.endswith('.gz'):
                    # Define the name of the decompressed file (removing the '.gz' suffix)
                    decompressed_file_name = file[:-3]
                    decompressed_file_path = os.path.join(timepoint_cd, decompressed_file_name)
                    
                    # Check if the decompressed file already exists
                    if os.path.isfile(decompressed_file_path):
                        # If it exists, add the original compressed file to the delete list
                        files_to_delete.add(os.path.join(timepoint_cd, file))
                    else:
                        # If not, decompress the '.gz' file using the gunzip command
                        os.system(f"gunzip {os.path.join(timepoint_cd, file)}")
    
    # Step 4: Print out the files that will be deleted
    print(f"Files to delete: {files_to_delete}")
    
    # Remove the files marked for deletion from the filesystem
    for file in files_to_delete:
        os.remove(file)
    
    # Indicate that the dataset processing is finished
    print("Dataset processing finished.")

if __name__ == "__main__":
    path = download_dataset()
    process_dataset(path)
    