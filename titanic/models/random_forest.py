import pandas as pd
import timeit
from sklearn.ensemble import RandomForestClassifier

def fit(train_data, test_data):
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X.to_numpy(), y.to_numpy())
    predictions = model.predict(X_test.to_numpy())

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    return output


def fit_pd(train_data, test_data):
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    return output


# Test classification running time with dataset as pandas DataFrame
def pd_time():
    SETUP_CODE = '''
from ..data import train_data, test_data
from __main__ import fit_pd
train = train_data()
test = test_data()
    '''
    TEST_CODE = '''
prediction = random_forest.fit(train, test)
    '''
    times = timeit.timeit(setup = SETUP_CODE, 
                          stmt = TEST_CODE,
                          number=100) 
    print("Random forest with pandas: {}".format(times))

# Test classification running time with dataset as numpy array
def np_time():
    SETUP_CODE = '''
from ..data import train_data, test_data
from __main__ import fit
train = train_data()
test = test_data()
    '''
    TEST_CODE = '''
prediction = fit(train, test)
    '''
    times = timeit.timeit(setup = SETUP_CODE, 
                          stmt = TEST_CODE,
                          number=100) 
    print("Random forest with numpy: {}".format(times))