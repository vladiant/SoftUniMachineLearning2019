import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

attributes, clusters = make_blobs()

inertias = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(attributes)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()
