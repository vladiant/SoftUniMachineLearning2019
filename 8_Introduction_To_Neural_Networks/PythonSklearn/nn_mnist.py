import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

# Data source notes
# https://www.tu-chemnitz.de/mathematik/numa/lehre/ds-2018/exercises/mnist_784.csv
# https://www.kaggle.com/oddrationale/mnist-in-csv/download
# http://yann.lecun.com/exdb/mnist/

mnist_data = pd.read_csv("../Data/mnist_784.zip")

print("Data loaded")

# Check initial data parameters
# print(mnist_data.shape)
# print(mnist_data.head())

# print(mnist_data.mean(axis=0))
# print(mnist_data.max(axis=0))
# print(mnist_data.loc[0])
# print(mnist_data.loc[0].to_numpy())

# number_data = mnist_data.loc[0].to_numpy()
# number_pixels = number_data[:-1]
# number = number_data[-1]

# print(number) #5

# plt.imshow(number_pixels.reshape(28, 28), cmap="Greys")
# plt.show()

attributes = mnist_data.drop("class", axis=1)
labels = mnist_data["class"]

# Check the data for training
# print(attributes.shape)
# print(labels.shape)

# Scaling
attributes /= 255

# print(attributes.mean(axis=0))

attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes, labels, train_size=0.8, stratify=labels
)

# Check the splitting
# print(attributes_train.shape)
# print(attributes_test.shape)
# print(labels_train.shape)
# print(labels_test.shape)

# nn = MLPClassifier(hidden_layer_sizes=(10,))

# Regularization - less units 3
# nn = MLPClassifier(hidden_layer_sizes=(3,))

# New hidden layer - 3, 3
# nn = MLPClassifier(hidden_layer_sizes=(3, 3))

# More hidden layers - 2, 3, 3, 2
nn = MLPClassifier(hidden_layer_sizes=(2, 3, 3, 2))

print("Start train")

nn.fit(attributes_train, labels_train)

print("Train score")
print(nn.score(attributes_train, labels_train))

print("Test score")
print(nn.score(attributes_test, labels_test))

# 10 units
# 0.9615
# 0.9352142857142857

# Regularization - less units 3
# 0.8288392857142857
# 0.8194285714285714

# New hidden layer - 3, 3
# 0.8215178571428572
# 0.8083571428571429

# More hidden layers - 2, 3, 3, 2
# 0.6590535714285715
# 0.6501428571428571
