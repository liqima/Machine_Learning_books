import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, (2, 3)] # petal length. petal width
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state = 42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])
# print(y_pred)

a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
b = -per_clf.intercept_ / per_clf.coef_[0][1]

axes = [0, 5, 0, 2]

x0, x1 = np.meshgrid(
	np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
	np.linspace(axes[2], axes[3], 200).reshape(-1, 1),)
X_new = np.c_[x0.ravel(), x1.ravel()]
y_predict = per_clf.predict(X_new)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize = (10, 4))
plt.plot(X[y == 0, 0], X[y == 0, 1], 'bs', label = 'Not Iris-Setosa')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'yo', label = 'Iris-Setosa')
plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], 'k-', linewidth = 3)

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#9898ff', '#fafab0'])
plt.contourf(x0, x1, zz, cmap = custom_cmap, linewidth = 5)
plt.xlabel('Petal length', fontsize = 14)
plt.ylabel('Petal width', fontsize = 14)
plt.legend(loc = 'lower right', fontsize = 14)
plt.axis(axes)

plt.show()