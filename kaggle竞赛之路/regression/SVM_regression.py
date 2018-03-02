from sklearn.datasets import load_boston

boston = load_boston()
#print(boston.DESCR) 


#
from sklearn.cross_validation import train_test_split
import numpy as np

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)

print('the max target value is ', np.max(y))
print('the min target value is ', np.min(y))
print('the average target value is ', np.mean(y))
print('\n')


# standard
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#y_train = ss.fit_transform(y_train)
#y_test = ss.transform(y_test)


# with support vector machine
from sklearn.svm import SVR
# 线性核函数
linear_svr = SVR(kernel = 'linear')
linear_svr.fit(x_train, y_train)
linear_y_pred = linear_svr.predict(x_test)
# 多项式核函数
poly_svr = SVR(kernel = 'poly')
poly_svr.fit(x_train, y_train)
poly_y_pred = poly_svr.predict(x_test)
# 径向积核函数
rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(x_train, y_train)
rbf_y_pred = rbf_svr.predict(x_test)


# report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('the R-squared value of linear SVR is ', linear_svr.score(x_test, y_test))
print('mean_squared_error of linear SVR is  ', mean_squared_error(y_test, linear_y_pred))
print('mean mean_absloute_error of liner SVR is ', mean_absolute_error(y_test, linear_y_pred))
print('\n')

print('the R-squared value of linear SVR is ', poly_svr.score(x_test, y_test))
print('mean_squared_error of linear SVR is  ', mean_squared_error(y_test, poly_y_pred))
print('mean mean_absloute_error of liner SVR is ', mean_absolute_error(y_test, poly_y_pred))
print('\n')

print('the R-squared value of linear SVR is ', rbf_svr.score(x_test, y_test))
print('mean_squared_error of linear SVR is  ', mean_squared_error(y_test, rbf_y_pred))
print('mean mean_absloute_error of liner SVR is ', mean_absolute_error(y_test, rbf_y_pred))