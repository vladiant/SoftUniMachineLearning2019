import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# SVR - regression
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt

boston_dataset = load_boston()

print(boston_dataset.data)
print(boston_dataset.target)
print(boston_dataset.feature_names)

features = pd.DataFrame(boston_dataset.data)
features.columns = boston_dataset.feature_names
print(features.head())

labels = pd.DataFrame(boston_dataset.target)
print(labels.head())

print(features.info())
print(labels.info())

print("features size ", features.shape)
print("labels size ", labels.shape)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.3
)

print("train features size ", train_features.shape)
print("train labels size ", train_labels.shape)
print("test features size ", test_features.shape)
print("test labels size ", test_labels.shape)

model = SVR(C=1e9)
model.fit(train_features, train_labels)

print("train score", model.score(train_features, train_labels))
print("test score", model.score(test_features, test_labels))

for column_label in features.columns:
    # print(column_label)
    plt.title(column_label)
    plt.scatter(test_features[column_label], test_labels, label="original data")
    plt.scatter(
        test_features[column_label], model.predict(test_features), label="fitted data"
    )
    # plt.scatter((min_x, min_y), (max_x, max_y), label = "fitted data")
    plt.legend()
    plt.show()
