from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:/Users/maliqi/Desktop/tensorflow/MNIST_data')

import tensorflow as tf
import numpy as np
from functools import partial
# construction phase
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape = [None, n_inputs])
he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
	activation = tf.nn.elu,
	kernel_initializer = he_init,
	kernel_regularizer = l2_regularizer)
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation = None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# exection phase
n_epochs = 5
batch_size = 150

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		n_batches = mnist.train.num_examples // batch_size
		for iteration in range(n_batches):
			# print('\r{}%'.format(100 * iteration // n_batches), end = '')
			# sys.stdout.flush()
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict = {X:X_batch})
		loss_train = reconstruction_loss.eval(feed_dict = {X:X_batch})
		print(epoch, ' train mse: ', loss_train)

