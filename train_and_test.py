#!/bin/python3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib

train_path = 'data/train_data.txt'
test_path = 'data/test_data.txt'


def generate_column_list():
    """Generate names of features."""
    pre_list = ['radius', 'texture',
                'perimeter', 'area',
                'smoothness', 'compactness',
                'concavity', 'concave-pts',
                'symmetry', 'fractal']

    suffix_list = ['mean', 'std', 'worst']

    return ['id', 'label'] + [pre + '_' + suffix for suffix in suffix_list for pre in pre_list]


def get_data(path):
    """Read the data and return it as features and labels"""
    data = pd.read_csv(path, header=None)
    columns = generate_column_list()
    data.columns = columns

    return data[columns[2:]], data[['label']]


if __name__ == '__main__':
    X_train, Y_train = get_data(train_path)
    X_test, Y_test = get_data(test_path)
    X_train, Y_train, X_test, Y_test = X_train.values, Y_train.values.ravel(), X_test.values, Y_test.values.ravel()

    Y_train = np.array(list(map(lambda x: 1 if x == 'B' else 0, Y_train)))
    Y_test = np.array(list(map(lambda x: 1 if x == 'B' else 0, Y_test)))

    model = xgb.XGBClassifier()

    print("Training the model")
    model.fit(X_train, Y_train)

    print("Saving the model")
    joblib.dump(model, 'model_save')

    print("Loading it back up")
    loaded_model = joblib.load('model_save')

    train_score = loaded_model.score(X_train, Y_train)
    test_score = loaded_model.score(X_test, Y_test)

    print("Train data score: %.3f" % train_score)
    print("Test data score: %.3f" % test_score)