from sklearn.datasets import load_boston

boston = load_boston()
#print(boston.DESCR) 


#
from sklearn.cross_validation import train_test_split
import numpy as np

x = boston.data
y = boston.target
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)

print('the max target value is ', np.max(y))
print('the min target value is ', np.min(y))
print('the average target value is ', np.mean(y))


# standard
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#y_train = ss.fit_transform(y_train)
#y_test = ss.transform(y_test)


# with linearRegression and SGDRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

lr = LinearRegression()
sr = SGDRegressor()

lr.fit(x_train, y_train)
sr.fit(x_train, y_train)

lr_y_predict = lr.predict(x_test)
sr_y_predict = sr.predict(x_test)



# report
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('the value of default measurement of LinearRegression is ',lr.score(x_test, y_test))
print('r squareed is ', r2_score(y_test, lr_y_predict))
print('mean squared error is ', mean_squared_error(y_test, lr_y_predict))
print('mean absoluate error is ', mean_absolute_error(y_test, lr_y_predict))


print('the value of default measurement of SGDRegression is ',sr.score(x_test, y_test))
print('r squareed is ', r2_score(y_test, sr_y_predict))
print('mean squared error is ', mean_squared_error(y_test, sr_y_predict))
print('mean absoluate error is ', mean_absolute_error(y_test, sr_y_predict))