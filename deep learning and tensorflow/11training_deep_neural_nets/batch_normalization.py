import tensorflow as tf
import numpy as np

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
training = tf.placeholder_with_default(False, shape = (), name = 'training')

# hidden1 = tf.layers.dense(X, n_hidden1, name = 'hidden1')
# bn1 = tf.layers.batch_normalization(hidden1, training = training, momentum = 0.9)
# bn1_act = tf.nn.elu(bn1)

# hidden2 = tf.layers.dense(bn1_act, n_hidden2, name = 'hidden2')
# bn2 = tf.layers.batch_normalization(hidden2, training = training, momentum = 0.9)
# bn2_act = tf.nn.elu(bn2)

# logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = 'outputs')
# logits = tf.layers.batch_normalization(logits_before_bn, training = \
# 	training, momentum = 0.9)

from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization, training = \
	training, momentum = 0.9)
hidden1 = tf.layers.dense(X, n_hidden1, name = 'hidden1')
bn1 = my_batch_norm_layer(hidden1)
bn1_act = tf.nn.elu(bn1)
hidden2 = tf.layers.dense(bn1_act, n_hidden2, name = 'hidden2')
bn2 = my_batch_norm_layer(hidden2)
bn2_act = tf.nn.elu(bn2)
logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name = 'outputs')
logits = my_batch_norm_layer(logits_before_bn)