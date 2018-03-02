import pandas as pd

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

x['age'].fillna(x['age'].mean(), inplace = True)


# split
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 33)


# feature extraction
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
x_train = vec.fit_transform(x_train.to_dict(orient = 'record'))
x_test = vec.transform(x_test.to_dict(orient = 'record'))


# with single decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)


# with random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)


# with gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)


# report
from sklearn.metrics import classification_report

print('decision tree: ', dtc.score(x_test, y_test))
print(classification_report(dtc_y_pred, y_test))

print('random forest: ', rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print('gradient boosting: ', gbc.score(x_test, y_test))
print(classification_report(gbc_y_pred, y_test))