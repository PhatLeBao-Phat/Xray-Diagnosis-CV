from tqdm import tqdm
import torch
import torch.nn as nn
from net import Net
from batch_sampler import BatchSampler
from typing import Callable, List
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import MulticlassF1Score, \
    MulticlassROC, MulticlassAUROC, MulticlassSpecificity, MulticlassPrecision, MulticlassRecall


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
        anomaly=False):
    # Lets keep track of all the losses:
    losses = []
    # initiate total labels and correct predictions for the accuracy
    total = 0
    correct = 0
    # initiate y pred and y true for confusion matrix
    y_pred = []
    y_true = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Set anomaly detection 
        # torch.autograd.detect_anomaly(anomaly)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # Get prediction probabilities
        y_pred_prob = nn.Softmax(dim=1)(predictions.data)
        # Get the predictions labels
        # predicted = torch.max(predictions.data, 1)[1]
        predicted = y_pred_prob.argmax(1)
        # Get the number of correct predictions
        total += y.size(0)
        correct += torch.sum(torch.eq(predicted, y)).item()
        # Save the true y and predicted y
        y = y.numpy()
        y_true.extend(y)
        y_pred.extend(predicted)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses, correct/total, confusion_matrix(y_true, y_pred),


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
        anomaly=False,
        ):
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    # initiate total labels and correct predictions for the accuracy
    total = 0
    correct = 0
    # initiate y pred and y true for the confusion matrix
    y_pred = []
    y_true = []

    predictions_prob = torch.Tensor([])

    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            # Set anomaly detection 
            # torch.autograd.detect_anomaly(anomaly)
            # Get Predictions:
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
            # Get the prediction probabilities
            y_pred_prob = nn.Softmax(dim=1)(prediction.data)
            # Get the predictions labels
            predicted = torch.max(prediction.data, 1)[1]
            # Get the number of correct predictions
            total += y.size(0)
            correct += torch.sum(torch.eq(predicted, y)).item()
            # Save the true y and predicted y
            y = y.numpy() # Inplace BUG-ALERT
            y_true.extend(y)
            y_pred.extend(predicted)
            predictions_prob = torch.cat((predictions_prob, y_pred_prob), 0)

    y_true_tensor = torch.Tensor(y_true).int()
    f1_score = get_f1_score(predictions_prob, y_true_tensor)
    roc = get_roc_curve(predictions_prob, y_true_tensor)
    auc_score = get_auc_score(predictions_prob, y_true_tensor)
    specificity = get_specificity(predictions_prob, y_true_tensor)
    precision = get_precision(predictions_prob, y_true_tensor)
    recall = get_recall(predictions_prob, y_true_tensor)

    return losses, correct/total, confusion_matrix(y_true, y_pred), f1_score.tolist(), roc, auc_score.tolist(), \
        specificity.tolist(), precision.tolist(), recall.tolist()


def get_f1_score(pred_prob, y_true):
    f1_score_metric = MulticlassF1Score(num_classes=6, average=None)
    f1_score = f1_score_metric(pred_prob, y_true)
    return f1_score


def get_roc_curve(pred_prob, y_true):
    roc_curve_metric = MulticlassROC(num_classes=6, thresholds=None)
    roc = roc_curve_metric(pred_prob, y_true)
    return roc


def get_auc_score(pred_prob, y_true):
    auc_score_metric = MulticlassAUROC(num_classes=6, thresholds=None, average=None)
    auc_score = auc_score_metric(pred_prob, y_true)
    return auc_score


def get_specificity(pred_prob, y_true):
    specificity_metric = MulticlassSpecificity(num_classes=6, average=None)
    specificity = specificity_metric(pred_prob, y_true)
    return specificity


def get_precision(pred_prob, y_true):
    precision_metric = MulticlassPrecision(num_classes=6, average=None)
    precision = precision_metric(pred_prob, y_true)
    return precision


def get_recall(pred_prob, y_true):
    recall_metric = MulticlassRecall(num_classes=6, average=None)
    recall = recall_metric(pred_prob, y_true)
    return recall
