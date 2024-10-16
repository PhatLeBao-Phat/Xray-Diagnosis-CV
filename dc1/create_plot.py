import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt 
import os

key_classes = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No Finding",
    4: "Nodule",
    5: "Pneumothorax"
}

def create_plot(nr_of_epochs, list_tester, metric_name,time_stamp):

    fig, ax = plt.subplots(figsize = (12,10),constrained_layout = True)   
    for i in range(0,6):
        ax.plot(range(1,nr_of_epochs+1),list_tester, label=i)
    ax.set_title(f'{metric_name} For Each Class Image On Test Data')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    
    # Set the tick locations
    ax.set_xticks(np.arange(0, nr_of_epochs+1, 1))

    # Legend part
    ax.legend(key_classes.values(), loc='lower left', bbox_to_anchor=(1.04, 0))

    # Display the plot
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))
    # fig.show()
    fig.savefig(Path("artifacts") / f"{metric_name}_{time_stamp.month:02}_{time_stamp.day:02}_{time_stamp.hour}_{time_stamp.minute:02}.png")