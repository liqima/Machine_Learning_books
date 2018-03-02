import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read the digits data using pandas
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)

# split 64 bit feature and 1 bit target 
x_train = digits_train[np.arange(64)] 
y_train = digits_train[64]
x_test = digits_test[np.arange(64)] 
y_test = digits_test[64]

# import KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10)
kmeans.fit(x_train)
y_pred = kmeans.predict(x_test)

# judge
from sklearn import metrics
print('ARI is ', metrics.adjusted_rand_score(y_test, y_pred))