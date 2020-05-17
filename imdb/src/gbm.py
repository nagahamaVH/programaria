import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import seaborn as sns

# Prepare data
from imdb.src.prepare_data_categorical import *

# Correction of levels for xgboost
y_train = pd.to_numeric(y_train) - 1
y_test = pd.to_numeric(y_test) - 1

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test, label=y_test)

# Tuning Hyperopt
space = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'max_depth': hp.quniform('max_depth', 3, 40, 1),
    'gamma': hp.uniform ('gamma', 1, 50),
    'reg_alpha': hp.quniform('reg_alpha', 10, 200, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 50, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'seed': 413}


def objective(space):
    clf = xgb.XGBClassifier(
        objective=space['objective'],
        num_class=space['num_class'],
        n_estimators=int(space['n_estimators']),
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
    print('SCORE: %.6f' % accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()

best_hyperparams = fmin(
    objective,
    space=space,
    algo=tpe.suggest,
    max_evals=250,
    trials=trials)

print(best_hyperparams)

# Fitting model with tuned hyperparameters
best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
best_hyperparams['objective'] = 'multi:softmax'
best_hyperparams['num_class'] = 4

gbm = xgb.XGBClassifier(best_hyperparams)

evaluation = [(x_train, y_train), (x_test, y_test)]

gbm.fit(x_train, y_train, eval_set=evaluation, early_stopping_rounds=10, 
        verbose=False)

pred = gbm.predict(x_test)

print(classification_report(y_test, pred))

conf_matrix = confusion_matrix(y_test, pred)

sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.show()