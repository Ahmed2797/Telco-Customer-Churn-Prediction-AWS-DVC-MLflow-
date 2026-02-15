import os 
import yaml 
import pickle
import joblib
import sys
import numpy as np
import pandas as pd
from project.logger import logging


def read_yaml(file_path: str):
    """
    Reads a YAML file and returns its content as a Python dictionary.
    
    Parameters:
        file_path (str): Path to the YAML file.
    
    Returns:
        dict: Parsed content of the YAML file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        
        if content is None:
            raise ValueError(f"YAML file is empty: {file_path}")
        
        # print(content['model_selection'].keys())  # ['module_0', 'module_1', 'module_2']
        
        return content

    except Exception as e:
        raise Exception(f"Error reading YAML file: {e}")



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a Python object (like dict or list) to a YAML file.
    
    Parameters:
        file_path (str): Path where YAML file will be written.
        content (object): Data (dict, list, etc.) to write into the YAML file.
        replace (bool): If True, replaces existing file. Default is False.
    
    Returns:
        None
    """
    try:
        # Remove existing file if replace=True
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            return yaml.dump(content, file, default_flow_style=False)

    except Exception as e:
        raise Exception(f"Error writing YAML file: {e}")
 
        


def save_object(file_path: str, obj) -> None:
    """
    Save any Python object (like a preprocessing pipeline or model)
    as a pickle (.pkl) file.

    Parameters
    ----------
    file_path : str
        The full file path where the object should be saved.
    obj : object
        The Python object to be saved.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Write the object using pickle
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise Exception(f"Error while saving object: {e}")





def save_numpy_array(file_path: str, array: np.ndarray) -> None:
    """
    Save a NumPy array to a .npy file.

    Args:
        file_path (str): Path where the array will be saved.
        array (np.ndarray): NumPy array to save.
    """
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the array
        np.save(file_path, array)
        print(f"Array saved successfully at: {file_path}")

    except Exception as e:
        print(f"Error saving array: {e}")





def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load a NumPy array from a .npy file.

    Args:
        file_path (str): Path to the saved array.

    Returns:
        np.ndarray: Loaded NumPy array
    """
    try:
        array = np.load(file_path)
        print(f"Array loaded successfully from: {file_path}")
        return array
    except Exception as e:
        print(f"Error loading array: {e}")
        return None


# def load_object(file_path: str) -> object:   
#     try:
#         with open(file_path, "rb") as file_obj:
#             obj = pickle.load(file_obj)

#         return obj
#     except Exception as e:
#         print(f"Error saving array: {e}")

def load_object(file_path: str) -> object:   
    try:
        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)

        return obj
    except Exception as e:
        print(f"Error saving array: {e}")



from statsmodels.stats.outliers_influence import variance_inflation_factor

def remove_high_vif_features(self, X: pd.DataFrame, threshold: float = 5.0):

    X_copy = X.copy()

    # Safety: convert bool to int
    bool_cols = X_copy.select_dtypes(include="bool").columns
    X_copy[bool_cols] = X_copy[bool_cols].astype(int)

    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_copy.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X_copy.values, i)
            for i in range(X_copy.shape[1])
        ]

        max_vif = vif_data["VIF"].max()

        if max_vif > threshold:
            drop_feature = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
            logging.info(f"Dropping '{drop_feature}' with VIF = {max_vif:.2f}")
            X_copy = X_copy.drop(columns=[drop_feature])
        else:
            break

    return X_copy
