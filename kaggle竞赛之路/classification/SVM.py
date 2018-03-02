from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape) 

# split the train dataset and test dataset
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)
print(y_train.shape)



# standard the data
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test= ss.transform(x_test)



# 
from sklearn.svm import LinearSVC

lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

print('accuracy if the support vector machine is ', lsvc.score(x_test, y_test))

#
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))