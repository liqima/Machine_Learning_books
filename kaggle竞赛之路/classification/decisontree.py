# obtain the dataset
import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#titanic.info()
#print(titanic.head())



# preprocessing
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

x['age'].fillna(x['age'].mean(), inplace = True)   # add data for age feature
#x.info()


# split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)


# feature extraction
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
#print(vec.feature_names_)
x_test = vec.transform(x_test.to_dict(orient = 'record'))


# import decision tree model
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict = dtc.predict(x_test)


# report
from sklearn.metrics import classification_report
print('the accuracy is ', dtc.score(x_test, y_test))
print(classification_report(y_predict, y_test, target_names = ['died', 'survived']))
