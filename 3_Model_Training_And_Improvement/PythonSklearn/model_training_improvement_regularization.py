import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt

iris_dataset = load_iris()

attributes = pd.DataFrame(iris_dataset.data[:, [2, 3]])
attributes.columns = iris_dataset.feature_names[:2]
print(attributes.head())

labels = pd.DataFrame(iris_dataset.target)
print(labels.head())

print(attributes.info())
print(labels.info())

print("attributes size ", attributes.shape)

weights, weights1, params = [], [], []

for c in range(-5, 5):
    model = LogisticRegression(C=10 ** c)
    model.fit(attributes, labels)
    weights.append(model.coef_[1]) # Display only second class
    weights1.append(model.coef_[0])
    params.append(10.0 ** c)

weights = np.array(weights)
weights1 = np.array(weights1)

print(weights1)

plt.plot(params, weights[:, 0], label=attributes.columns[0])
plt.plot(params, weights[:, 1], label=attributes.columns[1])
plt.xlabel("C")
plt.ylabel("Weight coefficient")
plt.xscale("log")
plt.legend()
plt.show()

plt.plot(params, weights1[:, 0], label=attributes.columns[0])
plt.plot(params, weights1[:, 1], label=attributes.columns[1])
plt.xlabel("C")
plt.ylabel("Weight coefficient")
plt.xscale("log")
plt.legend()
plt.show()
