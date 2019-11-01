import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# SVC - classification with kernel trick
from sklearn.svm import SVC
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

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, stratify=labels)

print("train features size ", train_features.shape)
print("train labels size ", train_labels.shape)
print("test features size ", test_features.shape)
print("test labels size ", test_labels.shape)

model = SVC(kernel='rbf', C=1e6, gamma=0.5)
model.fit(train_features, train_labels)

print("train score ", model.score(train_features, train_labels))
print("test score ", model.score(test_features, test_labels))

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