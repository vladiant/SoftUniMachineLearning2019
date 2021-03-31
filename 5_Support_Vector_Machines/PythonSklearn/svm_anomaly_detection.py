import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# SVR - regression
from sklearn.svm import OneClassSVM


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

anomaly_detector = OneClassSVM(nu=0.01, kernel="poly")
print(anomaly_detector)

anomaly_detector.fit(train_features)

# print(anomaly_detector.predict(train_features))
# print(anomaly_detector.predict(test_features))
print(
    anomaly_detector.predict(train_features)[
        anomaly_detector.predict(train_features) == 1
    ].sum()
    / train_features.shape[0]
)
print(
    anomaly_detector.predict(test_features)[
        anomaly_detector.predict(test_features) == 1
    ].sum()
    / test_features.shape[0]
)

for column_label in features.columns:
    # print(column_label)
    plt.title("test" + column_label)
    plt.scatter(
        test_features[column_label][anomaly_detector.predict(test_features) == 1],
        test_labels[anomaly_detector.predict(test_features) == 1],
        label="inlier data",
    )
    plt.scatter(
        test_features[column_label][anomaly_detector.predict(test_features) == -1],
        test_labels[anomaly_detector.predict(test_features) == -1],
        label="outlier data",
    )
    plt.legend()
    plt.show()

for column_label in features.columns:
    # print(column_label)
    plt.title("train" + column_label)
    plt.scatter(
        train_features[column_label][anomaly_detector.predict(train_features) == 1],
        train_labels[anomaly_detector.predict(train_features) == 1],
        label="inlier data",
    )
    plt.scatter(
        train_features[column_label][anomaly_detector.predict(train_features) == -1],
        train_labels[anomaly_detector.predict(train_features) == -1],
        label="outlier data",
    )
    plt.legend()
    plt.show()
