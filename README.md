## Code structure
The template code is structured into multiple files, based on their functionality. 
There are five `.py` files in total, each containing a different part of the code. 
Feel free to create new files to explore the data or experiment with other ideas.

- To download the data: run the `ImageDataset.py` file. The script will create a directory `/data/` and download the training and test data with corresponding labels to this directory. 
    - You will only have to run this script once usually, at the beginning of your project.

- To run the whole training/evaluation pipeline: run `main.py`. This script is prepared to do the followings:
    - Load your train and test data (Make sure its downloaded beforehand!)
    - Initializes the neural network as defined in the `Net.py` file.
    - Initialize loss functions and optimizers. If you want to change the loss function/optimizer, do it here.
    - Define number of training epochs and batch size
    - Check and enable GPU acceleration for training (if you have CUDA or Apple Silicon enabled device)
    - Train the neural network and perform evaluation on test set at the end of each epoch.
    - Provide plots about the training losses both during training in the command line and as a png (saved in the `/artifacts/` subdirectory)
    - Finally, save your trained model's weights in the `/model_weights/` subdirectory so that you can reload them later.

In your project, you are free to modify any parts of this code based on your needs. 
Note that the Neural Network structure is defined in the `Net.py` file, so if you want to modify the network itself, you can do so in that script.
The loss functions and optimizers are all defined in `main.py`.

## Environment setup instructions
We recommend to set up a virtual Python environment to install the package and its dependencies. To install the package, we recommend to execute `pip install -r requirements.txt.` in the command line. This will install it in editable mode, meaning there is no need to reinstall after making changes. If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on startup. Note that also in this case, it will still be necessary to run the pip command described above.

## Mypy
The template is created with support for full typehints. This enables the use of a powerful tool called `mypy`. Code with typehinting can be statically checked using this tool. It is recommended to use this tool as it can increase confidence in the correctness of the code before testing it. Note that usage of this tool and typehints in general is entirely up to the students and not enforced in any way. To execute the tool, simply run `mypy .`. For more information see https://mypy.readthedocs.io/en/latest/faq.html

## Argparse
Argparse functionality is included in the main.py file. This means the file can be run from the command line while passing arguments to the main function. Right now, there are arguments included for the number of epochs (nb_epochs), batch size (batch_size), and whether to create balanced batches (balanced_batches). You are free to add or remove arguments as you see fit.