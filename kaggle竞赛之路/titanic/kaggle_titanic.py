import pandas as pd


# read the train and test data file
train = pd.read_csv ('C:/Users/maliqi/Desktop/kaggle/train.csv')
test = pd.read_csv ('C:/Users/maliqi/Desktop/kaggle/test.csv')
#print(train.info())
#print(test.info())
#print(train.head())
#print(train.describe())     # object values and missed values are not included

#
import matplotlib.pyplot as plt

fig = plt.figure()

plt.subplot2grid((2, 3),(0, 0))
train.Survived.value_counts().plot(kind = 'bar')

plt.subplot2grid((2, 3), (0, 1))
train.Pclass.value_counts().plot(kind = 'bar')
plt.xlabel(u'class')

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(train.Survived, train.Age)
plt.grid(b = True, which = 'major', axis = 'y')

plt.subplot2grid((2, 3), (1, 0), colspan =2)
train.Age[train.Pclass == 1].plot(kind = 'kde')
train.Age[train.Pclass == 2].plot(kind = 'kde')
train.Age[train.Pclass == 3].plot(kind = 'kde')
plt.legend((u'1st', u'2nd', u'3rd'))

plt.subplot2grid((2, 3), (1, 2))
train.Embarked.value_counts().plot(kind = 'bar')

#plt.show()

# fill empty data
#print(x_train['Embarked'].value_counts())
#print(x_test['Embarked'].value_counts())
train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)
#x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
#print(x_test.info())

# missed age
from sklearn.ensemble import RandomForestRegressor
def missed_age(df):
	age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

	known_age = age_df[age_df.Age.notnull()].as_matrix()
	unknown_age = age_df[age_df.Age.isnull()].as_matrix()

	y = known_age[:,0]
	x = known_age[:, 1:]

	rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
	rfr.fit(x, y)

	predict_age = rfr.predict(unknown_age[:, 1:])
	df.loc[ (df.Age.isnull()), 'Age' ] = predict_age

	return df
train = missed_age(train)
test = missed_age(test)

def set_cabin(df):
	df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
	df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
	return df
train = set_cabin(train)
test = set_cabin(test)
# missed cabin

# select usful feature
#selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare', 'Cabin']
#x_train = train[selected_features]
x_train = train
y_train = train['Survived']
#x_test = test[selected_features]
x_test = test




# feature extraction
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
#print(dict_vec.feature_names_)
x_test = dict_vec.transform(x_test.to_dict(orient='record'))


# standard
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
'''
age_scale = ss.fit(x_train['Age'])
x_train['Age_scaled'] = ss.fit_transform(x_train['Age'], age_scale)

fare_scale = ss.fit(x_train['Fare'])
x_train['Fare_scaled'] = ss.fit_transform(x_train['Fare'], fare_scale)
'''
# import models
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier()

from xgboost import XGBClassifier
xgbc = XGBClassifier()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)

# validation
from sklearn.cross_validation import cross_val_score
print(cross_val_score(rfc, x_train, y_train, cv=5).mean())
print(cross_val_score(xgbc, x_train, y_train, cv=5).mean())
print(cross_val_score(dtc, x_train, y_train, cv=5).mean())
print(cross_val_score(knn, x_train, y_train, cv=5).mean())


# prediction
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)
rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_pred})
rfc_submission.to_csv('C:/Users/maliqi/Desktop/kaggle/rfc_submission.csv', index=False)

xgbc.fit(x_train, y_train)
xgbc_y_pred = xgbc.predict(x_test)
xgbc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':xgbc_y_pred})
xgbc_submission.to_csv('C:/Users/maliqi/Desktop/kaggle/xgbc_submission.csv', index=False)

dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)
dtc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':dtc_y_pred})
dtc_submission.to_csv('C:/Users/maliqi/Desktop/kaggle/dtc_submission.csv', index=False)

knn.fit(x_train, y_train)
knn_y_pred = knn.predict(x_test)
knn_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':knn_y_pred})
knn_submission.to_csv('C:/Users/maliqi/Desktop/kaggle/knn_submission.csv', index=False)