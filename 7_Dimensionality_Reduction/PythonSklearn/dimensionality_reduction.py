import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap

iris_dataset = load_iris()

features = pd.DataFrame(iris_dataset.data)
features.columns = iris_dataset.feature_names
print(features.head())

labels = pd.DataFrame(iris_dataset.target)
print(labels.head())

print(features.info())
print(labels.info())

print("features size ", features.shape)
print("labels size ", labels.shape)

iris_data = iris_dataset.data
iris_labels = iris_dataset.target
# print(iris_dataset.data)

pca = PCA()
pca.fit(iris_data)
iris_data_transformed = pca.transform(iris_data)

print(pca.explained_variance_ratio_)

print(iris_data_transformed.mean(axis=0))

iris_attributes = iris_data_transformed[:, :2]

# print(iris_attributes)

lr_normal = LogisticRegression(C=1e5).fit(iris_data, iris_labels)
# lr_transofrmed = LogisticRegression(C=1e5).fit(iris_attributes, iris_labels)
lr_transofrmed = LogisticRegression(C=1e5).fit(
    iris_attributes[:, 0].reshape(-1, 1), iris_labels
)

print(lr_normal.score(iris_data, iris_labels))
# print(lr_transofrmed.score(iris_attributes, iris_labels))
print(lr_transofrmed.score(iris_attributes[:, 0].reshape(-1, 1), iris_labels))


isomap = Isomap(n_neighbors=10, n_components=2)
iris_data_isomap = isomap.fit_transform(iris_data)

plt.scatter(iris_data_isomap[:, 0], iris_data_isomap[:, 1], c=iris_labels)
plt.show()

plt.scatter(iris_data_transformed[:, 0], iris_data_transformed[:, 1], c=iris_labels)
plt.show()
