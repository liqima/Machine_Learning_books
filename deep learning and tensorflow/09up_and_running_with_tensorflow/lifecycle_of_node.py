import tensorflow as tf

w = tf.constant(3)
x = w + 2
y = x + 5
z = y * 3

# with tf.Session() as sess:
# 	print(y.eval())
# 	print(z.eval())

with tf.Session() as sess:
	y_val, z_val = sess.run([y, z])
	print(y_val)
	print(z_val)