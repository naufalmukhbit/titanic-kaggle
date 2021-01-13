import numpy as np
import pandas as pd
import random

from copy import deepcopy
from .accuracy import calculate_accuracy

def fold(k, data):
    validation_data = data.copy()
    folds = []
    fold_size = validation_data.shape[0] // k
    for _ in range(k):
        fold = pd.DataFrame([])
        x = 1 if validation_data.shape[0] % fold_size > 0 else 0
        for __ in range(fold_size + x):
            pop_index = random.randrange(validation_data.shape[0])
            fold = fold.append(validation_data.iloc[pop_index], ignore_index=True)
            validation_data = validation_data.drop([pop_index]).reset_index(drop=True)
        folds.append(fold)
    return folds

def validate(k, data, model):
    folds = fold(k, data)
    scores = []
    for k in range(len(folds)):
        trainset = deepcopy(folds)
        testset = trainset.pop(k)
        trainset = pd.concat([i for i in trainset])
        prediction = model.fit(trainset, testset)
        scores.append(calculate_accuracy(prediction["Survived"], testset["Survived"]))
    return sum(scores) / len(scores)
