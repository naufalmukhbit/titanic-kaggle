import numpy as np
def calculate_accuracy(predicted, actual):
    '''
    Assume variables 'predicted' and 'actual' are numpy arrays, calculate the predicted accuracy towards the actual class
    '''
    return (np.array(predicted) == np.array(actual)).mean()