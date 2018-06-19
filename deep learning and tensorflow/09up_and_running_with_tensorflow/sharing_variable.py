import tensorflow as tf
import numpy as np

# def relu(X, threshold):
# 	with tf.name_scope('relu'):
# 		if not hasattr(relu, 'threshold'):
# 			relu.threshold = tf.Variable(0.0, name = 'threshold')
# 		w_shape = (int(X.get_shape()[1]), 1)
# 		w = tf.Variable(tf.random_normal(w_shape), name = 'weights')
# 		b = tf.Variable(0.0, name = 'bias')
# 		z = tf.add(tf.matmul(X, w), b, name = 'z')
# 		return tf.maximum(z, relu.threshold, name = 'max')

n_features = 3
threshold = tf.Variable(0.0, name = 'threshold')
X = tf.placeholder(tf.float32, shape = (None, n_features), name = 'X')
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name = 'output')
