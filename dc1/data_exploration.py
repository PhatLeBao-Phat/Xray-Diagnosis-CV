import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch

import image_dataset as im

# read test and train data
test_data = im.ImageDataset('data/X_test.npy', "data/Y_test.npy")
train_data = im.ImageDataset("data/X_train.npy", "data/Y_train.npy")
aug_train_data = im.ImageDataset("new_data/X_train.npy", "new_data/Y_train.npy")
# count number of data
nr_of_train_images = train_data.__len__()
nr_of_test_images = test_data.__len__()
nr_of_aug_train_images = aug_train_data.__len__()
nr_of_all_images = nr_of_train_images + nr_of_test_images
print(f"total number of images: {nr_of_all_images}")
print(f"number of test images: {nr_of_test_images}")
print(f"number of train images: {nr_of_train_images}")
print(f"number of augmented train images: {nr_of_aug_train_images}")

# count occurrences of each class
unique_labels_train, count_train = np.unique(train_data.targets, return_counts=True)
unique_labels_test, count_test = np.unique(test_data.targets, return_counts=True)
unique_aug_labels_train, aug_count_train = np.unique(aug_train_data.targets, return_counts=True)
unique_labels_all, count_all = np.unique(np.append(train_data.targets, test_data.targets), return_counts=True)

# Comparing the obtained plots with the plot in the project description,
# we can conclude the classes of the keys
key_classes = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax"
}

# Plot frequency of each case in the train set
plt.bar(list(map(key_classes.get, unique_labels_train)), count_train, color="r")
plt.bar(list(map(key_classes.get, unique_labels_test)), count_test, bottom=count_train, color="g")
plt.legend(["train", "test"])
plt.title("Frequency of each class in the train and test set")
plt.show()


# Plot frequency of each case in the test set
plt.bar(unique_labels_test, count_test)
plt.title("Frequency of each occurrence in the test set")
plt.show()

# Plot frequency of each case in all data
plt.bar(unique_labels_all, count_all)
plt.title("Frequency of each occurrence in all data")
plt.show()



# method to show the data of the train set with index i
def showDataTrainSet(i):
    image, label = train_data.__getitem__(i)
    plt.imshow(image[0], cmap="gray")
    plt.title(key_classes[label])
    plt.xticks([])
    plt.yticks([])
    plt.show()
# show an example of one instance of the data
showDataTrainSet(876)

print(40 * "-")
print("Exact number of occurrences for each class in the train set:")
for label, count in zip(unique_labels_train, count_train):
    print(f"{key_classes[label]}: {count}. {round((count / nr_of_train_images) * 100, 2)}%")

print(40 * "-")
print("Exact number of occurrences for each class in the test set:")
for label, count in zip(unique_labels_test, count_test):
    print(f"{key_classes[label]}: {count}. {round((count / nr_of_test_images) * 100, 2)}%")

print(40 * "-")
print("Exact number of occurrences for each class in all data:")
for label, count in zip(unique_labels_all, count_all):
    print(f"{key_classes[label]}: {count}. {round((count / nr_of_all_images) * 100, 2)}%")

print(40 * "-")
print("Exact number of occurrences for each class in augmented data:")
for label, count in zip(unique_aug_labels_train, aug_count_train):
    print(f"{key_classes[label]}: {count}. {round((count / nr_of_aug_train_images) * 100, 2)}%")