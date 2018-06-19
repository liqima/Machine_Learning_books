# fetch MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:\\Users\\maliqi\\Desktop\\tensorflow\\MNIST_data')

X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]
X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]

from dnn_classifier import DNNClassifier
dnn_clf = DNNClassifier(random_state = 42)
dnn_clf.fit(X_train1, y_train1, n_epochs=1000,
	X_valid=X_valid1, y_valid=y_valid1)