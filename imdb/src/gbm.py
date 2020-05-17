import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy import stats

# Prepare data
from imdb.src.prepare_data_categorical import *

# Correction of levels for xgboost
y_train = pd.to_numeric(y_train) - 1
y_test = pd.to_numeric(y_test) - 1

# Tuning Hyperopt
space = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform ('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': 180,
    'seed': 1254}


def objective(space):
    clf = xgb.XGBClassifier(
        objective=space['objective'],
        num_class=space['num_class'],
        n_estimators=space['n_estimators'],
        max_depth=int(space['max_depth']), 
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=space['reg_lambda'],
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=space['colsample_bytree'])

    evaluation = [(x_train, y_train), (x_test, y_test)]

    clf.fit(x_train, y_train, eval_set=evaluation, early_stopping_rounds=10,
            verbose=False)

    pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print('SCORE:', accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()

best_hyperparams = fmin(
    objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)
