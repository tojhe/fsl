# CM3070 Final Project - Code Repository

This is the code repository for the few-shot classifier implemented for the module UOL CS module
CM3070 Final Project.

In this repository:
- the `src` contains Python modules for training Prototypical Networks and Relation Networks.
- the Jupyter Notebooks document the machine learning workflow, flowing the sequence numbers in their filename prefixes.
- `pretrained_model` contains the relevant pickle file of the pretrained encoding module for Relation Networks from their original paper's code repository.
- `model` contains the pickles of best performing model parameter states of both algorithms after training. These can be loaded into state dictionaries and initialized into new models, for new users to use these models out of the box.
- `tensorboard` contains log files of all epoch's train/validation loss and accuracy during all trainings, and can be used for visualisation in Tensorboard.

The dataset used has been excluded in this repository due to its size. To gather the relevant dataset:
1. Down the dataset from this link to Kaggle: https://www.kaggle.com/datasets/gpiosenka/100-bird-species
2. Load the dataset into a folder `data/bird-classification` in the current directory level of this repository.
3. Run `0_eda.ipynb` to split the original dataset into few-shot learning usable train, validation and test datasets.

