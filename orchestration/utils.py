"""This module contains the utils functions for the orchestration
the aim of this module is to contain the functions that are used in the pipeline that are not
task or flow"""
import os
from config import DATA_PATH_PROCESSED
import pickle
import pandas as pd

def decode_labels_into_idx(labels: pd.Series, idx2label: dict) -> pd.Series:
    """This function decode the labels into idx
    Args:
      labels (pd.Series): series with the labels
      idx2label (dict): dictionary with the mapping
     Returns:
      labels (pd.Series): series with the labels decoded
    """
    return labels.map(idx2label)

def save_pickle(data, filename) -> None:
    """
    This function saves the data in a pickle file
    Args:
        data (object): data to save
        filename (str): filename
    Returns:
        None
    """
    filepath = os.path.join(DATA_PATH_PROCESSED, f"{filename}.pkl")
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(filename) -> object:
    """
    This function loads data from a pickle file.
    Args:
        filename (str): filename.
    Returns:
        data (object): data loaded from the pickle file.
    """
    filepath = os.path.join(DATA_PATH_PROCESSED, f"{filename}.pkl")
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data
