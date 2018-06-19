# Logistic Regression with Mini-Batch Gradient Descent using TensorFlow

# create the moons dataset
from sklearn.datasets import make_moons
m = 500
X_moons, y_moons = make_moons(m, noise = 0.1, random_state = 42)

# take a peek at the dataset
import matplotlib.pyplot as plt
# plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label = 'Positive')
# plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label = 'Negative')
# plt.legend()
# plt.show()

# add bias
import numpy as np
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
# print(X_moons_with_bias[:5])

# reshape y_train to make it a column vector
y_moons_cloumn_vector = y_moons.reshape(-1, 1)

# split the data into training set and test set
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_cloumn_vector[:-test_size]
y_test = y_moons_cloumn_vector[-test_size:]

# to generate training batches
def random_batch(X_train, y_train, batch_size):
	rnd_indices = np.random.randint(0, len(X_train), batch_size)
	X_batch = X_train[rnd_indices]
	y_batch = y_train[rnd_indices]
	return X_batch, y_batch


# build model
import tensorflow as tf
n_inputs = 2
learning_rate = 0.01

# construction phase
X = tf.placeholder(tf.float32, shape = (None, n_inputs + 1), name = 'X')
y = tf.placeholder(tf.float32, shape = (None, 1), name = 'y')
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed = 42, name = 'theta'))
logits = tf.matmul(X, theta, name = 'logits')
y_proba = tf.sigmoid(logits)
loss = tf.losses.log_loss(y, y_proba) # epsilon = 1e-1 by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epochs):
		for batch_index in range(n_batches):
			X_batch, y_batch = random_batch(X_train, y_train, batch_size)
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
		loss_val = loss.eval({X:X_test, y:y_test})
		if epoch % 100 == 0:
			print('Epoch: ', epoch, '/tLoss: ',loss_val)
	y_proba_val = y_proba.eval(feed_dict = {X:X_test, y:y_test})


y_pred = (y_proba_val >= 0.5)
from sklearn.metrics import precision_score, recall_score
print('precision: ' ,precision_score(y_test, y_pred))
print('recall: ', recall_score(y_test, y_pred))

y_pred_idx = y_pred.reshape(-1)
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label = 'Positive')
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label = 'Negative')
plt.legend()
plt.show()