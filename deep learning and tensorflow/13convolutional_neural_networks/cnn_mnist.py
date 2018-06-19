import tensorflow as tf
import numpy as np


height = 28
width = 28
channels = 1
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = 'SAME'

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = 'SAME'

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

with tf.name_scope('input'):
	X = tf.placeholder(tf.float32, shape = [None, n_inputs], name = 'X')
	X_reshaped = tf.reshape(X, shape = [-1, height, width, channels])
	y = tf.placeholder(tf.int32, shape = [None], name = 'y')

conv1 = tf.layers.conv2d(X_reshaped, filters = conv1_fmaps, 
	kernel_size = conv1_ksize, strides = conv1_stride, 
	padding = conv1_pad, activation = tf.nn.relu, name = 'conv1')
conv2 = tf.layers.conv2d(conv1, filters = conv2_fmaps, 
	kernel_size = conv2_ksize, strides = conv2_stride, padding = conv2_pad,
	activation = tf.nn.relu, name = 'conv2')

with tf.name_scope('pool3'):
	pool3 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
		padding = 'VALID')
	pool3_flat = tf.reshape(pool3, shape = [-1, pool3_fmaps * 7 * 7])

with tf.name_scope('fc1'):
	fc1 = tf.layers.dense(pool3_flat, n_fc1, activation = tf.nn.relu,
		name = 'fc1')

with tf.name_scope('output'):
	logits = tf.layers.dense(fc1, n_outputs, name = 'output')
	Y_proba = tf.nn.softmax(logits, name = 'Y_proba')

with tf.name_scope('train'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
		labels = y)
	loss = tf.reduce_mean(xentropy)
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('/tmp/data/')

def shuffle_batch(X, y, batch_size):
	rnd_idx = np.random.permutation(len(X))
	n_batches = len(X) // batch_size
	for batch_idx in np.array_split(rnd_idx, n_batches):
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		yield X_batch, y_batch

# load MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 5
batch_size = 100

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
		acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
		acc_test = accuracy.eval(feed_dict = {X:X_test, y:y_test})
		print(epoch, 'train acc: ', acc_train, 'acc test: ', acc_test)