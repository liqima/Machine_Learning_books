from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:/Users/maliqi/Desktop/tensorflow/MNIST_data')

import tensorflow as tf
import numpy as np
from functools import partial
import numpy.random as rnd

def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
                      learning_rate = 0.01, l2_reg = 0.0005, seed=42,
                      hidden_activation=tf.nn.elu,
                      output_activation=tf.nn.elu):
	graph = tf.Graph()
	with graph.as_default():
	    tf.set_random_seed(seed)

	    n_inputs = X_train.shape[1]

	    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
	    
	    my_dense_layer = partial(
	        tf.layers.dense,
	        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
	        kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))

	    hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
	    outputs = my_dense_layer(hidden, n_inputs, activation=output_activation, name="outputs")

	    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

	    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	    loss = tf.add_n([reconstruction_loss] + reg_losses)

	    optimizer = tf.train.AdamOptimizer(learning_rate)
	    training_op = optimizer.minimize(loss)

	    init = tf.global_variables_initializer()

	with tf.Session(graph=graph) as sess:
	    init.run()
	    for epoch in range(n_epochs):
	        n_batches = len(X_train) // batch_size
	        for iteration in range(n_batches):
	            indices = rnd.permutation(len(X_train))[:batch_size]
	            X_batch = X_train[indices]
	            sess.run(training_op, feed_dict={X: X_batch})
	        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
	        print("\r{}".format(epoch), "Train MSE:", loss_train)
	    params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
	    hidden_val = hidden.eval(feed_dict={X: X_train})
	    return hidden_val, params["hidden/kernel:0"], params["hidden/bias:0"], params["outputs/kernel:0"], params["outputs/bias:0"]
hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, 
	n_neurons=300, n_epochs=4, batch_size=150, output_activation=None)
_, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, 
	n_epochs=4, batch_size=150)

n_inputs = 28*28

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
outputs = tf.matmul(hidden3, W4) + b4
