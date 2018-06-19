

he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation = tf.nn.relu,
	kernel_initializer = he_init, name = 'hidden1')