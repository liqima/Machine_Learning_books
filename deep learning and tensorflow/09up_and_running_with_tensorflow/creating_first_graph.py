import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')
f = x * x * y + y + 2

# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()

# with tf.Session() as sess:
# 	x.initializer.run()
# 	y.initializer.run() # tf.get_default_session().run(x.initializer)
# 	result = f.eval()  # tf.get_default_session().run(f)
# print(result)

# init = tf.global_variables_initializer() # prepare an init node
# with tf.Session() as sess:
# 	init.run() # actually initialize all the Variables
# 	result = f.eval()
# print(result)

# init = tf.global_variables_initializer()
# sess = tf.InteractiveSession()
# init.run()
# result = f.eval()
# print(result)
# sess.close()

# x1 = tf.Variable(1)
# print(x1.graph is tf.get_default_graph())

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
	x2 = tf.Variable(2)
print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())