# CS6140FinalProject

# File structure

The original dataset of wav files and spectrograms is in the folder `Data/`.
A second copy of the wav files can be found in the `genres` directory.
Each of the `test/` `train/` `valid/` folders contain their respective datasets.

# Running the Code

All python files can be run directly from the command line.
`./preprocess.py` was used to generate the Mel Spectrograms and sperate into the Train, Validation, and test classes.
`./main.py` was used to train and score different models such as KNN and SVM using the sklearn package.
`./cnn.py` was used to train and score the CNN model. 
