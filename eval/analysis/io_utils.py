import os
import ast
from ruamel.yaml import YAML
import numpy as np

def load_params(params_fp):
    yaml = YAML(typ='safe')
    with open(params_fp, 'r') as f:
        params_temp = yaml.load(f)
    params = {}
    for key, val in params_temp.items():
        try:
            params[key] = ast.literal_eval(val)
        except:
            params[key] = val
    return params

def save_numpy_data(filepath, data):
    """Save numpy data to file."""
    np.save(filepath, data)

def load_numpy_data(filepath):
    """Load numpy data from file if exists."""
    if os.path.exists(filepath):
        return np.load(filepath)
    return None

def get_npy_files(folder_path):
    # List all .npy files in the folder
    npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    
    # Sort the files numerically based on their numeric part
    npy_files.sort(key=lambda x: int(x.split('.')[0]))
    
    return npy_files

def get_mat_files_in_range(data_dir, file_range):
    """
    Retrieves .mat file names within the specified range.

    Args:
        data_dir (str): Path to the directory containing .mat files.
        file_range [int, int]: Starting and ending number for subampling files.

    Returns:
        list: List of file names within the specified range.
    """
    # List all files in the directory
    all_files = os.listdir(data_dir)

    # Filter .mat files with numbers in the specified range
    filtered_files = []
    for file_name in all_files:
        if file_name.endswith('.mat'):
            try:
                # Extract number from the file name (assuming format like `123.mat`)
                number = int(os.path.splitext(file_name)[0])
                if file_range[0] <= number <= file_range[1]:
                    filtered_files.append(file_name)
            except ValueError:
                # Skip files that don't have a numeric name
                pass

    # Sort the files numerically based on their numeric part
    filtered_files.sort(key=lambda x: int(x.split('.')[0]))
    return filtered_files