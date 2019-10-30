# https://www.geeksforgeeks.org/ml-bagging-classifier/
from sklearn import model_selection
from sklearn.metrics import f1_score , classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

# dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
dataset = pd.read_csv("../Data/winequality-red.csv", sep=";")

# Check if values are balanced
print("Initial dataset class distribution")
print(dataset.quality.value_counts() * 100.0 /dataset.quality.count())
# Definitely not balanced

wine_features = dataset.drop("quality", axis=1)
wine_class = dataset.quality

wine_features_train, wine_features_test, wine_class_train, wine_class_test = model_selection.train_test_split(
    wine_features, wine_class, stratify=wine_class)

# Check sets distributions
# print(wine_features_train.shape)
# print(wine_features_test.shape)
# print(wine_class_train.shape)
# print(wine_class_test.shape)

print("Train dataset class distribution")
print(wine_class_train.value_counts() * 100.0 /wine_class_train.count())
print("Test dataset class distribution")
print(wine_class_test.value_counts() * 100.0 /wine_class_test.count())

seed = 8
kfold = model_selection.KFold(n_splits=5, random_state=seed)

# Initialize the base classifier
base_cls = DecisionTreeClassifier(random_state=seed)

base_results = model_selection.cross_val_score(base_cls, wine_features, wine_class, cv=kfold)
print("base accuracy: ", base_results.mean())

base_cls.fit(wine_features_train, wine_class_train)
print("Train base score: ", base_cls.score(wine_features_train, wine_class_train))
print("Test base score: ", base_cls.score(wine_features_test, wine_class_test))

print("Train base f1 score: ", f1_score(wine_class_train, base_cls.predict(wine_features_train), average='macro'))
print("Test base f1 score: ", f1_score(wine_class_test, base_cls.predict(wine_features_test), average='macro'))

print("Train base classification report:")
print(classification_report(wine_class_train, base_cls.predict(wine_features_train)))

print("Test base classification report:")
print(classification_report(wine_class_test, base_cls.predict(wine_features_test)))

# Number of base classifiers
num_trees = 500

# Bagging classifier
model = BaggingClassifier(base_estimator=base_cls, n_estimators=num_trees, random_state=seed)

results = model_selection.cross_val_score(model, wine_features, wine_class, cv=kfold)
print("accuracy: ", results.mean())

model.fit(wine_features_train, wine_class_train)
print("Train score: ", model.score(wine_features_train, wine_class_train))
print("Test score: ", model.score(wine_features_test, wine_class_test))
print("Train f1 score: ", f1_score(wine_class_train, model.predict(wine_features_train), average='macro'))
print("Test f1 score: ", f1_score(wine_class_test, model.predict(wine_features_test), average='macro'))

print("Train classification report:")
print(classification_report(wine_class_train, model.predict(wine_features_train)))

print("Test classification report:")
print(classification_report(wine_class_test, model.predict(wine_features_test)))
