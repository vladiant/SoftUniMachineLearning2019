import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression

# income_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=", ", header=None, engine = "python")
income_data = pd.read_csv("../Data/adult.data", sep=", ", header=None, engine="python")
print(income_data.head())

income_data.columns=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", \
                     "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",\
                     "income_class"]

income_data = income_data.drop("fnlwgt", axis=1)

income_data_attributes = income_data.drop("income_class", axis=1)
income_data_labels = income_data.income_class

# Scaling the data for numerical stability and accuracy
income_data_attributes = pd.get_dummies(income_data_attributes)
scaler = StandardScaler()
income_data_attributes_scaled = scaler.fit_transform(income_data_attributes)

# Split for training and testing
attributes_train, attributes_test, labels_train, labels_test = train_test_split(income_data_attributes_scaled, income_data_labels, train_size=0.7)

# Params for grid search
params = {
    "C" : [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000],
    "fit_intercept" : [True, False]
}
grid_search = GridSearchCV(LogisticRegression(), params, make_scorer(f1_score, pos_label=">50K"))
grid_search.fit(attributes_train, labels_train)

print("cv results:\n", grid_search.cv_results_)
print("best estimator: ", grid_search.best_estimator_)
print("best params: ", grid_search.best_params_)
print("best score: ", grid_search.best_score_)

# F1 score
best_model = grid_search.best_estimator_
train_predictions = best_model.predict(attributes_train)
test_predictions = best_model.predict(attributes_test)

print("train f1 score: ", f1_score(labels_train, train_predictions, pos_label=">50K"))
print("test f1 score: ", f1_score(labels_test, test_predictions, pos_label=">50K"))

