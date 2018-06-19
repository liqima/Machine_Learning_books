import tensorflow as tf
import numpy as np

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32,
	sequence_length = seq_length)

init = tf.global_variables_initializer()

X_batch = np.array([
	[[0, 1, 2], [9, 8, 7]],
	[[3, 4, 5], [0, 0, 0]],
	[[6, 7, 8], [6, 5, 4]],
	[[9, 0, 1], [3, 2, 1]]
	])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
	init.run()
	outputs_val = outputs.eval(feed_dict = {X:X_batch, 
		seq_length:seq_length_batch})
print(outputs_val)