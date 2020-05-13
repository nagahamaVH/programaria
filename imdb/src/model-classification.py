import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report

imdb = pd.read_csv('./imdb/data/categorical_imdb.csv')

y = imdb['categorical_imdb_score']
x = imdb.drop(['categorical_imdb_score'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3,
    random_state=42)

standardize = StandardScaler()
x_train = standardize.fit_transform(x_train)
x_test = standardize.transform(x_test)

logit = LogisticRegression(max_iter=1000)

#rfecv = RFECV(LogisticRegression(), cv=10, scoring='accuracy')

result = logit.fit(x_train, y_train)

y_pred = logit.predict(x_test)

cnf_matrix = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=['1', '2', '3', '4']))
