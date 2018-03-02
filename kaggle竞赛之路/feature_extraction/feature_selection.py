# obtain the dataset
import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#titanic.info()
print(titanic.head())


# preprocessing
x = titanic.drop(['row.names', 'name', 'survived'], axis=1)
y = titanic['survived']

x['age'].fillna(x['age'].mean(), inplace = True)   # add data for age feature
x.fillna('UNKNOWN', inplace=True)

# split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)


#feature extraction
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))
#print(len(vec.feature_names_))

# import decision tree model
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)
#y_predict = dtc.predict(x_test)
print(dtc.score(x_test, y_test))

from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
x_train_fs = fs.fit_transform(x_train, y_train)
dtc.fit(x_train_fs, y_train)
x_test_fs = fs.transform(x_test)
print(dtc.score(x_test_fs, y_test))