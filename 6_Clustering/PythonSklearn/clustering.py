# from sklearn.datasets import make_blobs , make_circles, make_regression, make_s_curve, make_swiss_roll, make_moons

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
attributes, clusters = make_blobs(cluster_std=1)

plt.scatter(attributes[:, 0], attributes[:, 1], c=clusters)
plt.show()

from sklearn.cluster import KMeans
attributes, clusters = make_blobs()

# Better n_init to be large!
k_means = KMeans(3, init="random", n_init=10)
assigned = k_means.fit_predict(attributes)

# Original, generated clusters
plt.scatter(attributes[:, 0], attributes[:, 1], c=clusters)
plt.show()

# Assigned clusters
plt.scatter(attributes[:, 0], attributes[:, 1], c=assigned)
plt.show()

k_means = KMeans(3, init="k-means++")
assigned = k_means.fit_predict(attributes)
assigned = k_means.fit_predict(attributes)

# Original, generated clusters
plt.scatter(attributes[:, 0], attributes[:, 1], c=clusters)
plt.show()

# Assigned clusters
plt.scatter(attributes[:, 0], attributes[:, 1], c=assigned)
plt.show()