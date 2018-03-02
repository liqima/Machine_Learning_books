import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
cluster4 = np.random.uniform(10.3, 10.2, (2, 10))


x = np.hstack((cluster1, cluster2, cluster3, cluster4)).T
#print(x)
plt.scatter(x[:,0], x[:,1]) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

K = range(1, 10)
meandistortions = []

for k in K:
	kmeans = KMeans(n_clusters = k)
	kmeans.fit(x)
	meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_, 'euclidean'), axis = 1))/x.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('average dispersion')
plt.title('selecting k with the elbow method')
plt.show()