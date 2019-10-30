import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

preset_iris_data = datasets.load_iris()

iris_data = pd.DataFrame(preset_iris_data.data)

iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

iris_data.append(pd.DataFrame(preset_iris_data.target))

X = preset_iris_data.data[:, :2]  # Sepal length, sepal width
y = preset_iris_data.target
h = 0.02  # Step size
color_dict = {0: "blue", 1: "lightgreen", 2: "red"}
colors = [color_dict[i] for i in y]
depth_2 = DecisionTreeClassifier(max_depth=2).fit(X, y)
depth_4 = DecisionTreeClassifier(max_depth=4).fit(X, y)
titles = ["Max depth = 2", "Max depth = 4"]

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for i, classifier in enumerate((depth_2, depth_4)):
    plt.figure()
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()
