# create the moons dataset
from sklearn.datasets import make_moons
m = 500
X_moons, y_moons = make_moons(m, noise = 0.1, random_state = 42)

# add bias
import numpy as np
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]

# reshape y_train to make it a column vector
y_moons_cloumn_vector = y_moons.reshape(-1, 1)

# split the data into training set and test set
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_cloumn_vector[:-test_size]
y_test = y_moons_cloumn_vector[-test_size:]

# add 4 more features
X_train_enhanced = np.c_[X_train,  np.square(X_train[:, 1]),
						np.square(X_train[:, 2]), X_train[:, 1] ** 3,
						X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test, np.square(X_test[:, 1]), 
						np.square(X_test[:, 2]), X_test[:, 1] ** 3,
						X_test[:, 2] ** 3]


# to generate training batches
def random_batch(X_train, y_train, batch_size):
	rnd_indices = np.random.randint(0, len(X_train), batch_size)
	X_batch = X_train[rnd_indices]
	y_batch = y_train[rnd_indices]
	return X_batch, y_batch

import tensorflow as tf

def logistic_regression(X, y, initializer = None, seed = 42, learning_rate = 0.01):
	n_inputs_including_bias = int(X.get_shape()[1])
	with tf.name_scope('logistic_regression'):
		with tf.name_scope('model'):
			if initializer is None:
				initializer = tf.random_uniform([n_inputs_including_bias, 1],
					-1.0, 1.0, seed = seed)
			theta = tf.Variable(initializer, name = 'theta')
			logits = tf.matmul(X, theta, name = 'logits')
			y_proba = tf.sigmoid(logits)
		with tf.name_scope('train'):
			loss = tf.losses.log_loss(y, y_proba, scope = 'loss')
			optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
			training_op = optimizer.minimize(loss)
			loss_summary = tf.summary.scalar('log_loss', loss)
		with tf.name_scope('init'):
			init = tf.global_variables_initializer()
		with tf.name_scope('save'):
			saver = tf.train.Saver()
	return y_proba, loss, training_op, loss_summary, init, saver

n_inputs = 2 + 4
X = tf.placeholder(tf.float32, shape = (None, n_inputs + 1), name = 'X')
y = tf.placeholder(tf.float32, shape = (None, 1), name = 'y')
y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)


n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
	start_epoch = 0
	sess.run(init)
	for epoch in range(start_epoch, n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
		loss_val, summary_str = sess.run([loss, loss_summary], feed_dict\
		 = {X:X_test_enhanced, y:y_test})
		if epoch % 100 == 0:
			print('Epoch: ', epoch, '/tLoss: ',loss_val)
	y_proba_val = y_proba.eval(feed_dict = {X:X_test_enhanced, y:y_test})


y_pred = (y_proba_val >= 0.5)
from sklearn.metrics import precision_score, recall_score
print('precision: ' ,precision_score(y_test, y_pred))
print('recall: ', recall_score(y_test, y_pred))

import matplotlib.pyplot as plt
y_pred_idx = y_pred.reshape(-1)
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label = 'Positive')
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label = 'Negative')
plt.legend()
plt.show()