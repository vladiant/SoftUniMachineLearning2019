import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report

from sklearn.tree import DecisionTreeClassifier

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
adult_income_data = adult_income_data.drop("fnlwgt", axis=1)

attributes = adult_income_data.drop("income", axis=1)
labels = adult_income_data.income

attributes = pd.get_dummies(attributes)
print(attributes.columns[0])

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

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(attributes_train, labels_train)


# Export tree as data
import pydotplus  # In Python3 instead pydot
import sklearn.tree as sklearn_tree
from sklearn.externals.six import StringIO

# dot_data = StringIO()
# sklearn_tree.export_graphviz(tree, out_file=dot_data)

dot_data = sklearn_tree.export_graphviz(tree, out_file=None, filled=True, rounded=True)

print(sklearn_tree.export_text(tree))
# print(type(dot_data))
graph = pydotplus.graph_from_dot_data(dot_data)

# graph = pydot.graph_from_dot_file("tree.dot")

graph.write_png("tree.png")

"""
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

im = np.array(Image(graph.create_png()))

plt.imshow(im)
plt.show()
"""

print(tree.score(attributes_train, labels_train))
print(tree.score(attributes_test, labels_test))
print(tree.get_depth())

params = {
    "max_depth": [2, 4, 10, 15, 20, 25, 28, 30],
    "min_samples_leaf": [5, 10, 15, 20, 30, 50],
}
grid = GridSearchCV(
    DecisionTreeClassifier(), params, scoring=make_scorer(f1_score, pos_label=" >50K")
)
grid.fit(attributes_train, labels_train)
print(grid.best_params_)
print(grid.best_score_)
# print(grid.cv_results_)

print("----")
print(grid.best_estimator_.score(attributes_train, labels_train))
print(grid.best_estimator_.score(attributes_test, labels_test))

# This metrics is to be used for comparison
predicted_labels_train = grid.best_estimator_.predict(attributes_train)
print(classification_report(labels_train, predicted_labels_train))

predicted_labels_test = grid.best_estimator_.predict(attributes_test)
print(classification_report(labels_test, predicted_labels_test))

# plt.bar(grid.best_estimator_.feature_importances_)

important_features_dict = {}
for x, i in enumerate(grid.best_estimator_.feature_importances_):
    important_features_dict[x] = i
    print(x, i)

important_features_list = sorted(
    important_features_dict, key=important_features_dict.get, reverse=True
)
for x in important_features_list:
    print(attributes.columns[x], important_features_dict[x])

importances = grid.best_estimator_.feature_importances_
indices = np.argsort(importances)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(
    range(attributes_train.shape[1]), importances[indices], color="r", align="center"
)
# If you want to define your own labels,
# change indices to a list of labels on the following line.
plt.yticks(range(attributes_train.shape[1]), indices)
plt.ylim([-1, attributes_train.shape[1]])
plt.show()
