import os
import pandas as pd

# Load the training data
# Path defaults to the 'train.py' file in the same folder
def train_data(path=os.path.join(os.path.dirname(__file__), "train.csv")):
    data = pd.read_csv(path)
    return data


# Load the testing data
# Path defaults to the 'test.py' file in the same folder
def test_data(path=os.path.join(os.path.dirname(__file__), "test.csv")):
    data = pd.read_csv(path)
    return data