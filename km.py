from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=200,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)

plt.figure(figsize=(6, 4), dpi=144)
plt.xticks(())
plt.yticks(())
plt.scatter(X[:, 0],X[:, 1], s=20, marker='o')
plt.show()

n_cluster = 3
k_means = KMeans(n_clusters=n_cluster)
k_means.fit(X)
print("kmean: k={}, cost={}".format(n_cluster, int(k_means.score(X))))

labels = k_means.labels_
centers = k_means.cluster_centers_

markers = ['o', '^', '*']
colors = ['r', 'b', 'y']

plt.figure(figsize=(6, 4), dpi=144)
plt.xticks(())
plt.yticks(())

for c in range(n_cluster):
    cluster = X[labels == c]
    plt.scatter(cluster[:, 0], cluster[:, 1], marker=markers[c], s=20, c=colors[c])

plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=0.9, s=300)
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % i, s=50, c=colors[i])

plt.show()
