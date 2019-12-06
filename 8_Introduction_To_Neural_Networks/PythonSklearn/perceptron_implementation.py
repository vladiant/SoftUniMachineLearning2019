# https://www.youtube.com/watch?v=QAOyXtfAI64&list=WL&index=93

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


def stepfunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepfunction((np.matmul(X, W) + b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
    for i in range(len(X)):
        y_pred = prediction(X[i], W, b)
        if (y[i] - y_pred == 1):
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        elif (y[i] - y_pred == -1):
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate

    return W, b


def trainPerceptronAlgo(X, y, learn_rate=0.01, num_epochs=25):
    x_max = max(X.T[0])
    W = np.array((np.random.rand(2, 1)))
    b = np.random.rand(1)[0] + x_max

    boundary_lines = []

    for i in range(num_epochs):
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))

    return boundary_lines


# data = pd.read_csv() # 100 elements, 2 coordinates, 1 label, it two b

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=((0.7, 0.7), (0.3, 0.3)), cluster_std=0.15)

print(X.shape)
print(y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y)
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.show()

result = trainPerceptronAlgo(X, y, 0.02, 100)

print(result)

plt.scatter(X[:, 0], X[:, 1], c=y)
vals = result[-1]
theta = vals[0][0]
intercept = vals[1][0]
y2 = X * theta + intercept

print(theta, intercept)

plt.plot(X, y2, color='red')
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.show()
