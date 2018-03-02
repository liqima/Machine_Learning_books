from sklearn.datasets import load_boston

boston = load_boston()


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

# two diffenent models
from sklearn.neighbors import KNeighborsRegressor
# 平均回归
uni_knr = KNeighborsRegressor(weights = 'uniform')
uni_knr.fit(x_train, y_train)
uni_knr_y_pred = uni_knr.predict(x_test)
# 根据距离加权回归
dis_knr = KNeighborsRegressor(weights = 'distance')
dis_knr.fit(x_train, y_train)
dis_knr_y_pred = dis_knr.predict(x_test)

# report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('the R-squared value of uni is ', uni_knr.score(x_test, y_test))
print('mean_squared_error of uni is  ', mean_squared_error(y_test, uni_knr_y_pred))
print('mean mean_absloute_error of uni is ', mean_absolute_error(y_test, uni_knr_y_pred))
print('\n')
print('the R-squared value of dis is ', dis_knr.score(x_test, y_test))
print('mean_squared_error of dis is  ', mean_squared_error(y_test, dis_knr_y_pred))
print('mean mean_absloute_error of dis is ', mean_absolute_error(y_test, dis_knr_y_pred))
print('\n')
