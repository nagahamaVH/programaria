import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

# Prepare data
from imdb.src.prepare_data_categorical import *

rfecv = RFECV(LogisticRegression(max_iter=1000), cv=10, scoring='accuracy', 
              n_jobs=-1)
rfecv.fit(x_train, y_train)

# Accuracy against number of features
grid_scores = rfecv.grid_scores_

plt.figure(figsize=(10, 6))
plt.xlabel('Number of features selected')
plt.ylabel('Accuracy')
plt.plot(range(1, len(grid_scores) + 1), grid_scores)
plt.plot(grid_scores.argmax(axis=0) + 1, max(grid_scores), 'o')
plt.show()

# Best features
print(rfecv.n_features_)
print(list(x.columns[rfecv.support_]))

x_train = x_train[:, rfecv.support_]
x_test = x_test[:, rfecv.support_] 

logit_model = LogisticRegression(max_iter=1000)
logit_model = logit_model.fit(x_train, y_train)

y_pred = logit_model.predict(x_test)

print(logit_model.score(x_test, y_test))
