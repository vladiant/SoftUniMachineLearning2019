# https://cambridgecoding.wordpress.com/2016/01/03/getting-started-with-regression-and-decision-trees/
# https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv
# https://gist.github.com/JustGlowing/fa2c0ac39415eb271db6

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import numpy as np

bikes = pd.read_csv("../Data/bikes.csv")
print(bikes.head())

# bikes.date = pd.to_numeric(pd.to_datetime(bikes.date))
# print(bikes.info())
# print(bikes.head())

plt.figure(figsize=(8, 6))
plt.plot(bikes["temperature"], bikes["count"], "o")
plt.xlabel("temperature")
plt.ylabel("bikes")
plt.show()

regressor = DecisionTreeRegressor(max_depth=2)
regressor.fit(np.array([bikes["temperature"]]).T, bikes["count"])

print(regressor.predict(np.array(5.0).reshape(-1, 1)))
print(regressor.predict(np.array(20.0).reshape(-1, 1)))

xx = np.array([np.linspace(-5, 40, 100)]).T

plt.figure(figsize=(8, 6))
plt.plot(bikes["temperature"], bikes["count"], "o", label="observation")
plt.plot(xx, regressor.predict(xx), linewidth=4, alpha=0.7, label="prediciton")
plt.xlabel("temperature")
plt.ylabel("bikes")
plt.show()

from sklearn.tree import export_graphviz

export_graphviz(regressor, out_file="tree.dot", feature_names=["temperature"])
dot_data = export_graphviz(regressor, out_file=None, feature_names=["temperature"])
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")
print(type(graph))

# https://stackoverflow.com/questions/19410042/how-to-make-ipython-notebook-matplotlib-plot-inline
# This should be first!
# %matplotlib inline
# from IPython.display import Image
# Image(filename='tree.png')
# Image(graph.create_png())

# https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/
# load and display an image with Matplotlib
import matplotlib.image as mpimg

# load image as pixel array
data = mpimg.imread("tree.png")
# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image
plt.imshow(data)
plt.show()
