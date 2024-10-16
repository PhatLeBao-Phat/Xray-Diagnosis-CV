from os import path
import os
import numpy as np
from matplotlib import pyplot as plt

# Load training data
X_train = np.load('data/X_train.npy')
Y_train = np.load('data/Y_train.npy')


def create_validation_set(X_train, Y_train, validation_ratio=0.2):
    """
    Splits the training data into a training set and a validation set.

    Parameters:
    X_train (numpy array): Training data features.
    Y_train (numpy array): Training data labels.
    validation_ratio (float): Ratio of training data to use for validation set.

    Returns:
    X_train_new (numpy array): Updated training data features, after removing validation set.
    Y_train_new (numpy array): Updated training data labels, after removing validation set.
    X_val (numpy array): Validation data features.
    Y_val (numpy array): Validation data labels.
    """
    # Shuffle the data to avoid any bias due to ordering
    np.random.seed(42)
    shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
    X_train_shuffled = X_train[shuffle_indices]
    Y_train_shuffled = Y_train[shuffle_indices]

    # Split the data into training and validation sets
    val_size = int(len(X_train) * validation_ratio)
    X_val = X_train_shuffled[:val_size]
    Y_val = Y_train_shuffled[:val_size]
    X_train_new = X_train_shuffled[val_size:]
    Y_train_new = Y_train_shuffled[val_size:]

    return X_train_new, Y_train_new, X_val, Y_val



