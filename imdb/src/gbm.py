import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report

# Prepare data
from imdb.src.prepare_data_categorical import *

# Correction of levels for xgboost
y_train = pd.to_numeric(y_train) - 1
y_test = pd.to_numeric(y_test) - 1

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test)

params = {
    'max_depth': 6,
    'objective': 'multi:softmax',
    'num_class': 4
}

gbm = xgb.train(params, train)

y_pred = gbm.predict(test)

print(classification_report(y_test, y_pred))
