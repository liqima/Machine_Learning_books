x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#
import numpy as np
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)
print(regressor.score(x_train, y_train))

# ploynomial regression   * 2
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree = 2)
x_train_poly2 = poly2.fit_transform(x_train)

regressor_poly2 = LinearRegression()
regressor_poly2.fit(x_train_poly2, y_train)

xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)
print(regressor_poly2.score(x_train_poly2, y_train))

# ploynomial regression   * 4
poly4 = PolynomialFeatures(degree = 4)
x_train_poly4 = poly4.fit_transform(x_train)

regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)

xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
print(regressor_poly4.score(x_train_poly4, y_train))


# plot
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train)

plt1, = plt.plot(xx, yy, label='degree=1')
plt2, = plt.plot(xx, yy_poly2, label='degree=2')
plt3, = plt.plot(xx, yy_poly4, label='degree=4')

plt.axis([0, 25, 0, 25])
plt.xlabel('diameter')
plt.ylabel('price')
plt.legend(handles=[plt1, plt2, plt3])
plt.show()


# report
x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

print(regressor.score(x_test, y_test))

# *2
x_test_poly2 = poly2.transform(x_test)
print(regressor_poly2.score(x_test_poly2, y_test))

# *4
x_test_poly4 = poly4.transform(x_test)
print(regressor_poly4.score(x_test_poly4, y_test))

# L1 regularization
from sklearn.linear_model import Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(x_train_poly4, y_train)
print(lasso_poly4.score(x_test_poly4, y_test))
print(lasso_poly4.coef_)
print(regressor_poly4.coef_)

# L2 regularization
from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(x_train_poly4, y_train)
print(ridge_poly4.score(x_test_poly4, y_test))