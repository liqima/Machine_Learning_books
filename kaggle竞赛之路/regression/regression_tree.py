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

# dicision tree regression
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_pred = dtr.predict(x_test)

# report
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('the R-squared value of dtr is ', dtr.score(x_test, y_test))
print('mean_squared_error of dtr is  ', mean_squared_error(y_test, dtr_y_pred))
print('mean mean_absloute_error of dtr is ', mean_absolute_error(y_test, dtr_y_pred))