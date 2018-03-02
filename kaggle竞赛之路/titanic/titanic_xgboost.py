import pandas as pd


# read the train and test data file
train = pd.read_csv ('C:/Users/maliqi/Desktop/kaggle/titanic/train.csv')
test = pd.read_csv ('C:/Users/maliqi/Desktop/kaggle/titanic/test.csv')
#print(train.info())
#print(test.info())
#print(train.head())


# select usful feature
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
x_train = train[selected_features]
y_train = train['Survived']
x_test = test[selected_features]


# fill empty data
#print(x_train['Embarked'].value_counts())
#print(x_test['Embarked'].value_counts())
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)
#print(x_test.info())


# feature extraction
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
#print(dict_vec.feature_names_)
x_test = dict_vec.transform(x_test.to_dict(orient='record'))


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


'''
# grid search
from sklearn.grid_search import GridSearchCV
params = {'max_depth':range(2, 7), 'n_estimators':range(100, 1100, 200), 'learning_rate':[0.05, 0.1, 0.25, 0.5, 1.0]}

xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params)
xgbc_best_y_pred = gs.predict(x_test)
xgbc_best_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':xgbc_best_y_pred})
xgbc_best_submission.to_csv('C:/Users/maliqi/Desktop/kaggle/xgbc_best_submission.csv', index=False)
'''