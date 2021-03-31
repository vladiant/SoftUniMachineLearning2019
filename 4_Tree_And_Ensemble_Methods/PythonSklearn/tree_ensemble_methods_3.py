import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# adult_income_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
adult_income_data = pd.read_csv("../Data/adult.data")
adult_income_data.columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

print(adult_income_data.columns)
print(adult_income_data.head())
# Remove curse
# adult_income_data = adult_income_data.drop("fnlwgt", axis=1)

attributes = adult_income_data.drop("income", axis=1)
labels = adult_income_data.income

attributes = pd.get_dummies(attributes)

scaler = MinMaxScaler()
# print(attributes.head())
# scaler.fit_transform(attributes)
# print(attributes.head())

attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes, labels, train_size=0.7, stratify=labels
)

print(attributes_train.shape)
print(attributes_test.shape)
print(labels_train.shape)
print(labels_test.shape)

pd.Series(labels_train).groupby(labels_train).size() / len(labels_train)

# Very weak algorithm
tree = DecisionTreeClassifier(max_depth=1)
tree.fit(attributes_train, labels_train)

print("Train Score")
print(tree.score(attributes_train, labels_train))
print("Test Score")
print(tree.score(attributes_test, labels_test))
print("Tree Depth")
print(tree.get_depth())

# This metrics is to be used for comparison
print("----")
print("train scores")
predicted_labels_train = tree.predict(attributes_train)
print(classification_report(labels_train, predicted_labels_train))
print("f1 score train")
print(f1_score(labels_train, predicted_labels_train, pos_label=" >50K"))

print("test scores")
predicted_labels_test = tree.predict(attributes_test)
print(classification_report(labels_test, predicted_labels_test))
print("f1 score test")
print(f1_score(labels_test, predicted_labels_test, pos_label=" >50K"))

params = {
    "n_estimators": [10, 100, 200, 250, 300],
    "max_depth": [2, 4, 10, 15, 20, 25, 28, 30],
}

# Use pickle here!
grid = GridSearchCV(
    RandomForestClassifier(max_depth=1),
    params,
    scoring=make_scorer(f1_score, pos_label=" >50K"),
)

grid.fit(attributes_train, labels_train)
print(grid.best_params_)
print(grid.best_score_)
# print(grid.cv_results_)

print("----")
print("train score")
print(grid.best_estimator_.score(attributes_train, labels_train))
print("test score")
print(grid.best_estimator_.score(attributes_test, labels_test))

# This metrics is to be used for comparison
print("train scores")
predicted_labels_train = grid.best_estimator_.predict(attributes_train)
print(classification_report(labels_train, predicted_labels_train))

print("test scores")
predicted_labels_test = grid.best_estimator_.predict(attributes_test)
print(classification_report(labels_test, predicted_labels_test))

"""
Unlimited depth
0.9846437346437347
0.8594389844389845
              precision    recall  f1-score   support

       <=50K       0.98      1.00      0.99     17303
        >50K       0.99      0.95      0.97      5489

    accuracy                           0.98     22792
   macro avg       0.99      0.97      0.98     22792
weighted avg       0.98      0.98      0.98     22792

              precision    recall  f1-score   support

       <=50K       0.88      0.94      0.91      7416
        >50K       0.76      0.61      0.68      2352

    accuracy                           0.86      9768
   macro avg       0.82      0.78      0.79      9768
weighted avg       0.85      0.86      0.85      9768
"""
