
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read the digits data using pandas
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)

# split 64 bit feature and 1 bit target 
x_digits = digits_train[np.arange(64)] 
y_digits = digits_train[64]

from sklearn.decomposition import PCA

estimator = PCA(n_components = 2)
x_pca = estimator.fit_transform(x_digits)

def plot_pca_scatter():
	colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
	for i in range(len(colors)):
		px = x_pca[:,0][y_digits.as_matrix()==i]
		py = x_pca[:,1][y_digits.as_matrix()==i]
		plt.scatter(px, py, c = colors[i])
	plt.legend(np.arange(0, 10).astype(str))
	plt.xlabel('first')
	plt.ylabel('second')
	plt.show()
plot_pca_scatter()

'''

import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home = 'C:/Users/maliqi/Desktop/mnist/')
x = mnist['data']
print(x.shape)

from sklearn.decomposition import PCA
estimator = PCA(n_components = 0.95)
x_pca = estimator.fit_transform(x)
print(x_pca.shape)

#print(estimator.explained_variance_ratio_)

#print(sum(estimator.explained_variance_ratio_))


