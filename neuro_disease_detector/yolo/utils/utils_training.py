import os

def generate_yaml_files(data_path: str) -> list:
    """
    Generate YAML configuration files for k-fold cross-validation.

    This function creates a list of dictionary configurations for a k-fold
    cross-validation setup. Each configuration specifies the training, validation, 
    and test dataset paths for one fold. The generated configurations can be used 
    in machine learning workflows to evaluate models across different folds.

    Args:
        data_path (str): The base path to the dataset directory.

    Returns:
        list: A list of dictionaries, where each dictionary represents the configuration
              for one fold. Each dictionary contains the following keys:
              - 'train': A list of paths to the training datasets (excluding validation and test folds).
              - 'test': A string representing the path to the test dataset for the fold.
              - 'val': A string representing the path to the validation dataset for the fold.
              - 'nc': The number of classes in the dataset (set to 1 by default).
              - 'names': A list containing the name(s) of the class(es) (default is ['multiple_esclerosis']).

    Example:
        data_path = '/path/to/dataset'
        yaml_configs = generate_yaml_files(data_path)
        for config in yaml_configs:
            print(config)

    Note:
        - The function assumes the dataset is organized in a directory structure where
          each fold's images are located under the path:
          `{data_path}/MSLesSeg-Dataset-a/foldX/images`, where `X` is the fold number (1 through k).
        - By default, `k` is set to 5 for 5-fold cross-validation.
    """

    k = 5

    folds = [f'fold{i}' for i in range(1, k+1)]
    kfold_configs = []
    
    # Iterate over each fold to create train, val, test splits
    for i in range(k):
        data = {
            'train': [],
            'test': '',
            'val': '',
            'nc': 1,
            'names': ['multiple_esclerosis']
        }
        
        # Set the test fold (current fold)
        data['test'] = os.path.join(data_path, f'MSLesSeg-Dataset-a/{folds[i]}/images')
        
        # Set the validation fold (next fold, wrapping around)
        val_fold = (i + 1) % k
        data['val'] = os.path.join(data_path, f'MSLesSeg-Dataset-a/{folds[val_fold]}/images')
        
        # Set the train folds (all except current and validation fold)
        for j in range(k):
            if j != i and j != val_fold:
                fold_path = os.path.join(data_path, f'MSLesSeg-Dataset-a/{folds[j]}/images')
                data['train'].append(fold_path)
        
        # Append this iteration's configuration to the list
        kfold_configs.append(data)
    
    # Return the YAML configurations for all iterations
    return kfold_configs
