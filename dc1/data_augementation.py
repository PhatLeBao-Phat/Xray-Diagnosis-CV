from typing import List, Tuple
import os.path as path
from pathlib import Path
import os
from collections import Counter
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class NewDataset(Dataset):
    """
    Represent a new Dataset object.
    """

    def __init__(self, data, targets, transform=None) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, np.ndarray]:
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            x = self.wtf(x)

        return torch.Tensor(x), y

    def __len__(self) -> int:
        return len(self.data)

    def wtf(self, x, inplace=True) -> torch.Tensor:
        """
        Transform the dataset of with image transformations and augementations.

        Data need to be converted to PIL first. Some of the transformations require PIL object instead of torch.Tensor. Note that our data only have one channel. Some have default=3 thus errors. Do the conversion here and call to other methods to avoid confusion.

        Parameters
        ----------
        x : np.adrray or torch.tensor, optional
            Image data (not targets/labels)

        Returns
        -------
        torch.Tensor

        See Also
        --------
        https://pytorch.org/vision/stable/transforms.html#conversion-transforms
        https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
        """
        x = self.transform((torch.Tensor(x)))

        return torch.Tensor(x)

    def __len__(self):
        return len(self.data)


def create_save_augmentation():
    # Load training data
    origin_train_data = np.load('data/X_train.npy')
    origin_train_targets = np.load('data/Y_train.npy')

    # Split the dataset into features (images) and labels
    X = origin_train_data
    y = origin_train_targets

    # Apply under-sampling
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X.reshape(X.shape[0], -1), y)

    # Reshape the under-sampled data to its original shape
    X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2], X.shape[3])


    # Define image transformations
    transform = T.Compose([
        T.RandomResizedCrop(32, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0)),
        T.RandomRotation(degrees=(-15, 15)),
        T.Resize((128, 128))
    ])

    # Init under-sampled data without transformation
    undersampled_dataset = NewDataset(
        X_resampled,
        y_resampled
    )

    # Transform under-sampled data
    transformed_dataset = NewDataset(
        X_resampled,
        y_resampled,
        transform=transform
    )

    # Select a subset of the augmented dataset
    augmented_data = list(zip(transformed_dataset.data, transformed_dataset.targets))
    random.shuffle(augmented_data)  # Shuffle the data
    n_augmented_samples = len(augmented_data)
    augmented_subset = augmented_data[:n_augmented_samples]

    # Unpack the subset into separate arrays for data and targets
    augmented_data_subset, augmented_target_subset = zip(*augmented_subset)

    #Concatenate original dataset with subset of transformed and oversampled dataset
    aug_X_train = np.concatenate((np.array(origin_train_data), np.array(augmented_data_subset)))
    aug_Y_train = np.concatenate((np.array(origin_train_targets), np.array(augmented_target_subset)))

    # Save arrays to npy file
    # check if new_data/ subdir exists
    if not Path("new_data/").exists():
        os.mkdir(Path("new_data/"))
    else:
        print("Data will be overwritten!")

    np.save("new_data/X_train.npy", aug_X_train)
    np.save("new_data/Y_train.npy", aug_Y_train)