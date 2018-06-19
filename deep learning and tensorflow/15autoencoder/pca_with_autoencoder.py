import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# build dataset
import numpy.random as rnd
rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * rnd.random(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.random(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m) 

# normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

# build autoencoder
n_inputs = 3
n_hidden = 2
n_outputs = n_inputs
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape = [None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

# execution phase
n_iterations = 300
codings = hidden
with tf.Session() as sess:
	init.run()
	for iteration in range(n_iterations):
		training_op.run(feed_dict = {X:X_train})
	codings_val = codings.eval(feed_dict = {X:X_test})

fig = plt.figure(figsize = (4, 3))
plt.plot(codings_val[:, 0], codings_val[:, 1], 'b.')
plt.xlabel('z1', fontsize = 18)
plt.ylabel('z2', fontsize = 18, rotation = 0)
plt.show()

