import torch
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
from net import PreTrainedAlexNet

from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset

test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=False
    )

model = PreTrainedAlexNet(6)
state_dict = torch.load("model_weights/model_04_05_19_46.pth")
model.load_state_dict(state_dict)
model.eval()

# We need to make sure we do not update our model based on the test data:
with torch.no_grad():
    correct = 0
    count = 0
    y_pred = []
    y_true = []
    for (x, y) in tqdm(test_sampler):
        # Making sure our samples are stored on the same device as our model:
        x = x.to("cpu")
        y = y.to("cpu")
        prediction = model.forward(x).argmax(axis=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(prediction.cpu().numpy())
        correct += sum(prediction == y)
        count += len(y)
        # cf_matrix = confusion_matrix(y_true, y_pred)

# 0 for no finding, 1 for disease
y_true_bin = [3 if x == 3 else -1 for x in y_true]
y_pred_bin = [3 if x == 3 else -1 for x in y_pred]

cf_matrix = confusion_matrix(y_true_bin, y_pred_bin, normalize="all")
print(cf_matrix)
labels = {3: 'No Findings', -1: 'Disease'}
plt.figure(figsize=(6, 6))
ax = sn.heatmap(cf_matrix, annot=True, vmin=0, vmax=1, xticklabels=["No Findings", "Disease"], yticklabels=["No Findings", "Disease"])
ax.set_title('Binary confusion matrix', size=20, weight='bold')
ax.set_ylabel('True', size=20)
ax.set_xlabel('Predicted', size=20)
plt.show()