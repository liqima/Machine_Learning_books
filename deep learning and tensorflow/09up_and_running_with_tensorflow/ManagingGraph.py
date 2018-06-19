import tensorflow as tf


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x1 = tf.Variable(1)
# print(x1.graph is tf.get_default_graph())

# # manage independent graph
# graph = tf.Graph()
# with graph.as_default():
# 	x2 = tf.Variable(2)
# print(x2.graph is graph)
# print(x2.graph is tf.get_default_graph())

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
# with tf.Session() as sess:
# 	print(y.eval())
# 	print(z.eval())

with tf.Session() as sess:
	y_val, z_val = sess.run([y, z])
	print(y_val)
	print(z_val)