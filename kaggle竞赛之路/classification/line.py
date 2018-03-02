import numpy as np
import pandas as pd

column_names = ['Sample code number', 'Clump thickness', 'Uniformity of cell size', 
				'Uniformity of cell shape', 'Marginal adheion', 'Single epithelial cell size', 
				'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitises', 'class']
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

print(data.shape)
#print(column_names[:2])

#分割数据成训练集+测试集
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], 
	test_size=0.25, random_state=33)
#print(y_train.value_counts())
#print(y_test.value_counts())

# use logistic regress model to classificite
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lr = LogisticRegression()
sgdc = SGDClassifier()
#to train the model
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)


# report
from sklearn.metrics import classification_report

print('accuracy of LR classification is ', lr.score(x_test, y_test))
print('', classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignan']))

print('accuracy of LR classification is ', sgdc.score(x_test, y_test))
print('', classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignan']))



