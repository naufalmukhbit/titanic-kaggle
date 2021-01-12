import numpy as np
import pandas as pd
import timeit

from titanic.data import train_data, test_data
from titanic.models import random_forest

train = train_data()
test = test_data()

prediction = random_forest.fit(train, test)
print(prediction)