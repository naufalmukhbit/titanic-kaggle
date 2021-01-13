import numpy as np
import random

from copy import deepcopy
from .accuracy import calculate_accuracy

def fold(k, data):
    validation_data = np.copy(data)
    folds = []
    fold_size = validation_data.shape[0] // k
    for _ in range(k):
        fold = []
        x = 1 if data.shape[0] % fold_size > 0 else 0
        for __ in range(fold_size + x):
            pop_index = random.randrange(validation_data.shape[0])
            fold.append(validation_data[pop_index].tolist())
            validation_data = np.delete(validation_data, pop_index, 0)
        fold = np.array(fold)
        folds.append(fold)
    return folds

def validate(k, data, model):
    folds = fold(k, data)
    scores = []
    for k in range(len(folds)):
        trainset = deepcopy(folds)
        testset = trainset.pop(k)
        trainset = np.concatenate(tuple([i for i in trainset]))
        prediction = model.fit(trainset, testset)
        scores.append(calculate_accuracy(prediction, trainset[:,1]))
    return sum(scores) / len(scores)
