import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

# income_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", sep=", ", header=None, engine = "python")
income_data = pd.read_csv("../Data/adult.data", sep=", ", header=None, engine="python")
print(income_data.head())

income_data.columns=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", \
                     "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",\
                     "income_class"]
print(income_data.head())

# plt.hist(income_data.fnlwgt)
plt.show()

income_data = income_data.drop("fnlwgt", axis=1)

# Check if set is balanced
print(income_data.groupby("income_class").size()/len(income_data))
# Not balanced
# income_class
# <=50K    0.75919
# >50K     0.24081
# dtype: float64

income_data_attributes = income_data.drop("income_class", axis=1)
income_data_labels = income_data.income_class

# Check shapes
print("income_data_attributes", income_data_attributes.shape)
print("income_data_labels", income_data_labels.shape)

# Input data types
print("types income_data_attributes", income_data_attributes.dtypes)
# workclass         object
# education         object
# education-num      int64
# marital-status    object
# occupation        object
# relationship      object
# race              object
# sex               object
# capital-gain       int64
# capital-loss       int64
# hours-per-week     int64
# native-country    object

# We need to have only numbers for processing
income_data_attributes = pd.get_dummies(income_data_attributes)
print("dummies income_data_attributes", income_data_attributes.shape)
print("head income_data_attributes", income_data_attributes.head())

# Description
print("description income_data_attributes\n", income_data_attributes.describe().T)

# Info
print("description income_data_attributes\n", income_data_attributes.info())

# Scaling the data for numerical stability and accuracy
scaler = StandardScaler()
income_data_attributes_scaled = scaler.fit_transform(income_data_attributes)

# Testing transformation with random
test_trasform = np.random.randint(1, 100, (50, 107))
print(test_trasform)
print(scaler.transform(test_trasform))

# Model for testing
logistic_model = LogisticRegression(C=1e6)
logistic_model.fit(income_data_attributes_scaled, income_data_labels)
print("all data scaled score ", logistic_model.score(income_data_attributes_scaled, income_data_labels))
logistic_model.fit(income_data_attributes, income_data_labels)
print("all data score ", logistic_model.score(income_data_attributes, income_data_labels))

# Split for training and testing
attributes_train, attributes_test, labels_train, labels_test = train_test_split(income_data_attributes_scaled, income_data_labels, train_size=0.7)

# Check the shapes
print("attributes_train ", attributes_train.shape)
print("attributes_test ", attributes_test.shape)
print("labels_train ", labels_train.shape)
print("labels_test ", labels_test.shape)

model = LogisticRegression(C=100)
model.fit(attributes_train, labels_train)
print("train score ", model.score(attributes_train, labels_train))
print("test score", model.score(attributes_test, labels_test))

# F1 score
train_predictions = model.predict(attributes_train)
test_predictions = model.predict(attributes_test)

print("train f1 score: ", f1_score(labels_train, train_predictions, pos_label=">50K"))
print("test f1 score: ", f1_score(labels_test, test_predictions, pos_label=">50K"))
