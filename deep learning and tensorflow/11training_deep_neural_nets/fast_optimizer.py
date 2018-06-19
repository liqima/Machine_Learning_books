import tensorflow as tf

# momentum optimization
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,
	momentum = 0.9)

# Nesterov Accelerated Gradient
optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate,
	momentum = 0.9, use_nesterov = True)

# AdaGrad
optimizer = tf.train.AdagradeOptimizer(learning_rate = learning_rate)

# RMSProp
optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate,
	momentum = 0.9, decay = 0.9, epsilon = 1e-10)

# Adam
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
