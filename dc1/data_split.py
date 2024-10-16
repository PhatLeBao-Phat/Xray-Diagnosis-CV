import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from sklearn.model_selection import StratifiedShuffleSplit

def split_validation_test(random_seed=42):
    """
    Split to train-validation-test with stratified sampling. The proportion is based on total
    data (combine both given training + test). Division is as following:
    - test 20% or 5053 samples
    - training 48% or 12124 samples
    - validation 32% or 8084 samples  
    Stratified dataset is saved into strat_data/
    
    Parameters
    -------
    random_seed: int, optional, default=42
        random generator for suffling
    
    Returns
    -------
    None

    Warns
    -------
    The random_seed is 42 (random_state)
    """        
    # PATH
    path_X_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\data\X_train.npy")
    path_X_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\data\X_test.npy")
    path_Y_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\data\Y_train.npy")
    path_Y_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\data\Y_test.npy")

    # Load 
    X_train = np.load(path_X_train)
    X_test = np.load(path_X_test)
    y_train = np.load(path_Y_train)
    y_test = np.load(path_Y_test)
    
    # Concatenate train and test set 
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))

    # Stratified split training and test set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    for train_index, test_index in split.split(X, y):
        strat_X_train = X[train_index]
        strat_X_test = X[test_index]
        strat_y_train = y[train_index]
        strat_y_test = y[test_index]

    # Stratified split training and validation 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    for train_index, valid_index in split.split(strat_X_train, strat_y_train):
        strat_X_train = X[train_index]
        strat_X_valid = X[valid_index]
        strat_y_train = y[train_index]
        strat_y_valid = y[valid_index]

    # Save path 
    save_path_X_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\strat_data\X_train.npy")
    save_path_X_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\strat_data\X_test.npy")
    save_path_Y_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\strat_data\Y_train.npy")
    save_path_Y_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\strat_data\Y_test.npy")
    save_path_X_valid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\strat_data\X_valid.npy")
    save_path_Y_valid = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..\strat_data\Y_valid.npy")
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../strat_data/")

    # Save file 
    if os.path.exists(save_path):
        np.save(save_path_X_test, strat_X_test)
        np.save(save_path_X_train, strat_X_train)
        np.save(save_path_X_valid, strat_X_valid)
        np.save(save_path_Y_test, strat_y_test)
        np.save(save_path_Y_train, strat_y_train)
        np.save(save_path_Y_valid, strat_y_valid)
    
    print('----------------------------------------------------------------')
