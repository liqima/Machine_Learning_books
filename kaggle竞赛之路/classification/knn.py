from sklearn.datasets import load_iris

iris = load_iris()
#print(iris)
print(iris.data.shape)
#print (iris.DESCR)

# split test/train dataset
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)


# standard
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# import the knn model
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)

# report
from sklearn.metrics import classification_report
print('the accuracy is ', knc.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names = iris.target_names))