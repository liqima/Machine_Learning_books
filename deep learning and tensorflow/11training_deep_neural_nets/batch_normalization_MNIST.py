import tensorflow as tf
import numpy as np
from functools import partial

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
batch_norm_momentum = 0.9
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')
training = tf.placeholder_with_default(False, shape = (), name = 'training')

with tf.name_scope('dnn'):
	he_init = tf.contrib.layers.variance_scaling_initializer()

	my_batch_norm_layer = partial(
		tf.layers.batch_normalization, 
		training = training,
		momentum = batch_norm_momentum)

	my_dense_layer = partial(
		tf.layers.dense,
		kernel_initializer = he_init)

	hidden1 = my_dense_layer(X, n_hidden1, name = 'hidden1')
	bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
	hidden2 = my_dense_layer(bn1, n_hidden2, name = 'hidden2')
	bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
	logits_before_bn = my_dense_layer(bn2, n_outputs, name = 'outputs')
	logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = 'loss')

with tf.name_scope('train'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 20
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# load MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_example // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run([training_op, extra_update_ops], 
				feed_dict = {training : True, X : X_batch, y : y_batch})
		accuracy_val = accuracy.eval(feed_dict = {X:mnist.test.images,\
			y:mnist.test.labels})
		print(epoch, "test acc: ", accuracy_val)