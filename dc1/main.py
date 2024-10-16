# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset, TRANSFORM
from net import Net, AlexNet, MODEL
from train_test import train_model, test_model
from create_plot import create_plot, key_classes
from data_split import split_validation_test

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
from captum.attr import Saliency
from scipy.ndimage import gaussian_filter

# import data augmentation file
import data_augementation

# Key classes
key_classes = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax"
}


# Prints the exact metrics and plot it
# Change plotext to plt to plot in new window
def print_metric_all_classes(performance_list, performance_name):
    print(35 * "-")
    print(f"{performance_name} per class: ")
    for i in range(6):
        print(f"{key_classes[i]}: {performance_list[i]}")

    plotext.clf()
    plotext.bar(key_classes.values(), performance_list)
    plotext.title(f"{performance_name}")
    # plotext.show()


# Method that plots the roc curve and displays the AUC
def plot_roc(roc, auc_score, e):
    # Check if /roc_curve/ subdir exists
    if not Path("roc_curve/").exists():
        os.mkdir(Path("roc_curve/"))

    fpr, tpr, threshold = roc
    plt.clf()
    plt.plot([0, 1], [0, 1], 'k-')
    for i in range(6):
        plt.plot(fpr[i], tpr[i], label=f"{key_classes[i]} AUC = {auc_score[i]}")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc='best')
    plt.savefig(Path("roc_curve") / f'epoch {e + 1}.jpg')
    # plt.show()
    plt.close()

def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # transform and save new data
    data_augementation.create_save_augmentation()

    # Load the train and test data set
    # train_dataset = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))
    train_dataset = ImageDataset(Path("new_data/X_train.npy"), Path("new_data/Y_train.npy"))
    test_dataset = ImageDataset(Path("data/X_test.npy"), Path("data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    try:
        model = MODEL[args.model_choice](n_classes=6)
    except:
        model = MODEL[args.model_choice]
    # fetch transformation techniques 
    trans_technique = args.transform
    if args.model_choice == 'GoogLeNet':
        trans_technique = 'GoogLeNet'
    train_dataset.transform = TRANSFORM[trans_technique]
    test_dataset.transform = TRANSFORM[trans_technique]
    # Fetch learing rate, weight decay, momentum from arguments 
    lr = args.lr
    wdecay = args.wd
    momentum = args.momentum

    # set initial weights computed in the data exploration file
    weights = torch.Tensor([0.14638658, 0.15920646, 0.12450762, 0.06046872, 0.22598933, 0.2834413])

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wdecay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)
    loss_function = nn.CrossEntropyLoss()
    # Fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size
    # Fetch input size 
    input_size = (1, args.input_size, args.input_size)
    if args.model_choice == 'GoogLeNet':
        input_size = (1, 224, 224)
    # IMPORTANT! Set this to True to see actual errors regarding
    DEBUG = False
    ANOMALY_DEBUG = args.debug_anomaly
    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, input_size, device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, input_size, device=device) # NOTICE CHANGE

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    mean_accuracy_train = []
    mean_accuracy_test = []
    f1_scores_test = []
    auc_score_test = []
    specificity_test = []
    precision_test = []
    recall_test = []


    # Check if /salience_map/ subdir exists
    if not Path("saliency_map/").exists(): # CHANGE
        os.mkdir(Path("saliency_map/"))

    for e in range(n_epochs):
        if activeloop:

            # Training:
            losses_train, accuracy_train, cf_matrix_train,= train_model(model,
                                                                        train_sampler,
                                                                        optimizer,
                                                                        loss_function,
                                                                        device,
                                                                        anomaly=ANOMALY_DEBUG,)

            # scheduler.step(sum(losses_train) / len(losses_train))

            # Saliency map:
            images, labels = next(iter(train_sampler))
            image = images[0]
            saliency = Saliency(model)
            image.requires_grad = False
            attribution = saliency.attribute(image.unsqueeze(0), target=labels[0])
            
            # Convert the saliency map to a numpy array
            saliency_map = attribution.permute(0, 2, 3, 1).detach().numpy()
            
            # Threshold the saliency map to focus on important areas
            threshold = 0.15
            thresholded_map = np.where(saliency_map > threshold, saliency_map, 0)
            thresholded_map = (thresholded_map - thresholded_map.min()) / (thresholded_map.max() - thresholded_map.min() + 1e-10)
            
            # Smooth the saliency map with a Gaussian filter
            thresholded_map = gaussian_filter(thresholded_map, sigma=2)
            
            # Plot saliency map
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
            ax1.imshow(image.permute(1, 2, 0).numpy())
            ax1.set_title(f"Label: {key_classes[labels[0].item()]}")
            ax2.imshow(thresholded_map[0], cmap='seismic')
            ax2.set_title("Saliency Map")
            ax3.imshow(image.permute(1, 2, 0).numpy())
            ax3.set_title(f"Label: {key_classes[labels[0].item()]}")
            saliency_map = thresholded_map[0, :, :, 0]
            ax3.imshow(saliency_map, cmap='seismic', alpha=0.4)
            ax3.set_title("Combined")
            plt.savefig(Path("saliency_map") / f'epoch {e + 1}.jpg')
            plt.close()


            # Calculating and printing statistics:
            mean_loss = sum(losses_train) / len(losses_train)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")
            mean_accuracy_train.append(accuracy_train)
            print(f"\nEpoch {e + 1} training done, accuracy on train set: {accuracy_train * 100}%\n")
            
            # Testing:
            losses_test, accuracy_test, cf_matrix_test, \
                f1_score_test, roc, auc_score, specificity, precision, recall = test_model(model,
                                                                                        test_sampler,
                                                                                        loss_function, 
                                                                                        device,
                                                                                        anomaly=ANOMALY_DEBUG,)

            # Calculating and printing statistics and performance metrics:
            mean_loss = sum(losses_test) / len(losses_test)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")
            print(f"\nEpoch {e + 1} testing done, accuracy on test set: {accuracy_test * 100}%\n")

            # plot ROC for this epoch
            plot_roc(roc, auc_score, e)

            # Display specificity and precision of each class
            print_metric_all_classes(f1_score_test, "F1 score")
            print_metric_all_classes(specificity, "Specificity")
            print_metric_all_classes(precision, "Precision")
            print_metric_all_classes(recall, "Recall")

            # Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()
            plotext.show()

            # Keep track of metrics
            f1_scores_test.append(f1_score_test)
            mean_accuracy_test.append(accuracy_test)
            auc_score_test.append(auc_score)
            specificity_test.append(specificity)
            precision_test.append(precision)
            recall_test.append(recall)

            # Plot confusion matrix during training
            classes = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Nodule", "Pneumothorax"]
            df_cm_train = pd.DataFrame(cf_matrix_train / np.sum(cf_matrix_train, axis=1), index=[i for i in classes],
                                       columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sns.heatmap(df_cm_train, annot=True)
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            plt.title('Confusion matrix for train set')
            # plt.show()

            # cf_matrix_test = confusion_matrix_test(model, train_sampler, device)
            df_cm_test = pd.DataFrame(cf_matrix_test / np.sum(cf_matrix_test, axis=1), index=[i for i in classes],
                                      columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sns.heatmap(df_cm_test, annot=True)
            plt.xlabel('Predicted label', fontsize=14)
            plt.ylabel('True label', fontsize=14)
            plt.title('Confusion matrix for test set')
            # plt.show()

    # Retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    
    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.pth")
    
    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    # # Create plot of accuracy
    # width = 0.4
    # ind = np.arange(16)
    #
    # fig, ax = plt.subplots()
    # bar_train = ax.bar(ind, [x for x in mean_accuracy_train], width, color='orange')
    # bar_test = ax.bar(ind+width, [x for x in mean_accuracy_test], width, color='blue')
    # ax.set_ylabel('Accuracy')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'))
    # ax.legend((bar_train[0], bar_test[0]), ('Train set', 'Test set'))
    # ax.set_ylim(0, 0.45)
    # for bars in ax.containers:
    #     ax.bar_label(bars, fmt='%.2f')

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of accuracy
    # fig.savefig(Path("artifacts") / f"accuracy_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")

    ####################################################################################
    ## Metrics plots
    # f1_scores_test.append(f1_score_test)
    # mean_accuracy_test.append(accuracy_test)
    # auc_score_test.append(auc_score)
    # specificity_test.append(specificity)
    # precision_test.append(precision)
    # recall_test.append(recall)
                                             
    create_plot(n_epochs,f1_scores_test,"F1 Score",now)
    create_plot(n_epochs,mean_accuracy_test,"Mean Accuracy",now)
    create_plot(n_epochs,auc_score_test,"Auc-Score",now)   
    create_plot(n_epochs,specificity_test,"Specificity",now)
    create_plot(n_epochs,precision_test,"Precision",now)
    create_plot(n_epochs,recall_test,"Recall",now)


if __name__ == "__main__":

    if not Path("strat_data/").exists():
        print("Create stratified data directory!")
        os.mkdir(Path("strat_data/"))
        split_validation_test()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=16, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=False,
        type=bool,
    )
    parser.add_argument("--model_choice", help="choose the model", default='TrainedAlexNet', type=str)
    parser.add_argument("--debug_anomaly", help="anomaly detection",default=False, type=bool)
    parser.add_argument("--lr", help="learning rate", default=0.001, type=float)
    parser.add_argument("--wd", help="weight decay", default=0.00, type=float)
    parser.add_argument("--momentum", help="momentum", default=0.9, type=float)
    parser.add_argument("--input_size", help="input_size for summary", default=128, type=int)
    parser.add_argument("--transform", help="transforms method for model", default='None', type=str)
    args = parser.parse_args()

    main(args)
