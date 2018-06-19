import tensorflow as tf

x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')
f = x * x * y + y + 2

# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)

# with tf.Session() as sess:
# 	x.initializer.run()
# 	y.initializer.run()
# 	result = f.eval()

init = tf.global_variables_initializer()
with tf.Session() as sess:
	init.run()
	result = f.eval()
 
# sess = tf.InteractiveSession()
# init.run()
# result = f.eval()

print(result)
# sess.close()