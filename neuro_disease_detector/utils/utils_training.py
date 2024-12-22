import os
import yaml

def generate_yaml_files(data_path: str) -> list:
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
