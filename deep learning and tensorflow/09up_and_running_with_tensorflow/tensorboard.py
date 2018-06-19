import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 500
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
ss = StandardScaler()
housing_data_plus_bias = ss.fit_transform(housing_data_plus_bias)

X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name = 'X')
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = 'y')
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = 'theta')
y_pred = tf.matmul(X, theta, name = 'predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = 'mse')
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

from datetime import datetime
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf.logs'
logdir = '{}/run-{}/'.format(root_logdir, now)

mse_summary = tf.summary.scalar('mse', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
file_writer.close()


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("epoch: ", epoch, 'MSE: ', mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()
print(best_theta)