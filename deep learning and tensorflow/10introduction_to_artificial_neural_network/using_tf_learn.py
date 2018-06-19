from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:/Users/maliqi/Desktop/tensorflow/10introduction_to_artificial_neural_network/MNIST_data/')

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')

import tensorflow as tf
config = tf.contrib.learn.RunConfig(tf_random_seed = 42)
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = [300, 100], n_classes = 10,
		feature_columns = feature_cols, config = config)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size = 50, steps = 1000)

y_pred = dnn_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred['classes']))

from sklearn.metrics import log_loss
y_pred_proba = y_pred['probabilities']
print(log_loss(y_test, y_pred_proba))