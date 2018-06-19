import tensorflow as tf
import numpy as np

# Construction Phase
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')

# def neuron_layer(X, n_neurons, name, activation = None):
# 	with tf.name_scope(name):
# 		n_inputs = int(X.get_shape()[1])
# 		stddev = 2 / np.sqrt(n_inputs)
# 		init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
# 		w = tf.Variable(init, name = 'weights')
# 		b = tf.Variable(tf.zeros([n_neurons]), name = 'biases')
# 		z = tf.matmul(X, w) + b
# 		if activation == 'relu':
# 			return tf.nn.relu(z)
# 		else:
# 			return z

# with tf.name_scope('dnn'):
# 	hidden1 = neuron_layer(X, n_hidden1, 'hidden1', activation = 'relu')
# 	hidden2 = neuron_layer(hidden1, n_hidden2, 'hidden2', activation = 'relu')
# 	logits = neuron_layer(hidden2, n_outputs, 'outputs')

from tensorflow.contrib.layers import fully_connected
with tf.name_scope('dnn'):
	hidden1 = fully_connected(X, n_hidden1, scope = 'hidden1')
	hidden2 = fully_connected(hidden1, n_hidden2, scope = 'hidden2')
	logits = fully_connected(hidden2, n_outputs, scope = 'outputs', \
		activation_fn = None)

with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels = y, logits = logits)
	loss = tf.reduce_mean(xentropy, name = 'loss')

learning_rate = 0.01
with tf.name_scope('train'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution Phase
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('/MNIST_data/')
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28)/255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28)/255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) // batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch

# with tf.Session() as sess:
# 	init.run()
# 	for epoch in range(n_epochs):
# 		for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
# 			sess.run(train_op, feed_dict = {X:X_batch, y:y_batch})
# 		acc_batch = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
# 		acc_val = accuracy.eval(feed_dict = {X:X_valid, y:y_valid})
# 		print(epoch, 'batch accuracy: ', acc_batch, 'val accuracy: ', acc_val)
# 	save_path = saver.save(sess, './my_model_final.ckpt')

with tf.Session() as sess:
	saver.restore(sess, './my_model_final.ckpt')
	X_new_scaled = X_test[:20]
	Z = logits.eval(feed_dict = {X:X_new_scaled})
	y_pred = np.argmax(Z, axis = 1)
print('predicted classes: ', y_pred)
print('actual classes: ', y_test[:20])

