# https://hackernoon.com/gradient-boosting-and-xgboost-90862daa6c77

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

iris_dataset = load_iris()

# print(iris_dataset.data)
# print(iris_dataset.target_names)
# print(iris_dataset.target)
# print(iris_dataset.feature_names)

features = pd.DataFrame(iris_dataset.data)
features.columns = iris_dataset.feature_names
print(features.head())

labels = pd.DataFrame(iris_dataset.target)
print(labels.head())

print(features.info())
print(labels.info())

print("features size ", features.shape)
print("labels size ", labels.shape)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.3, stratify=labels
)

print("train features size ", train_features.shape)
print("train labels size ", train_labels.shape)
print("test features size ", test_features.shape)
print("test labels size ", test_labels.shape)

#  learning_rate=0.1, n_estimators=100, loss='deviance',
model = GradientBoostingClassifier()
model.fit(train_features, train_labels)

print("train score", model.score(train_features, train_labels))
print(classification_report(train_labels, model.predict(train_features)))
print("test score", model.score(test_features, test_labels))
print(classification_report(test_labels, model.predict(test_features)))

"""
train score 1.0
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        37
           1       1.00      1.00      1.00        35
           2       1.00      1.00      1.00        33

    accuracy                           1.00       105
   macro avg       1.00      1.00      1.00       105
weighted avg       1.00      1.00      1.00       105

test score 0.9333333333333333
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       0.88      0.93      0.90        15
           2       0.94      0.88      0.91        17

    accuracy                           0.93        45
   macro avg       0.94      0.94      0.94        45
weighted avg       0.93      0.93      0.93        45
"""

X = iris_dataset.data[:, :2]  # Sepal length, sepal width
y = iris_dataset.target
h = 0.02  # Step size
color_dict = {0: "blue", 1: "lightgreen", 2: "red"}
colors = [color_dict[i] for i in y]

model.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

plt.figure()
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel(iris_dataset.feature_names[0])
plt.ylabel(iris_dataset.feature_names[1])
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks()
plt.yticks()
plt.show()
