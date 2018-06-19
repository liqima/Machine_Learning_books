import tensorflow as tf
import numpy as np

def reset_graph(seed = 42):
	tf.reset_default_graph()
	tf.set_random_seed(seed)
	np.random.seed(seed)


# build a dnn with five hidden layers of 10 neurons each
he_init = tf.contrib.layers.variance_scaling_initializer()

def dnn(inputs, n_hidden_layers = 5, n_neurons = 100, name = None,
	activation = tf.nn.elu, initializer = he_init):
	with tf.variable_scope(name, 'dnn'):
		for layer in range(n_hidden_layers):
			inputs = tf.layers.dense(inputs, n_neurons, activation = activation,
				kernel_initializer = initializer, name = 'hidden%d' % (layer + 1))
		return inputs

n_inputs = 28 * 28 # mnist
n_outputs = 5

reset_graph()

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')

dnn_outputs = dnn(X)

logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer = he_init,
		name = 'logits')
Y_proba = tf.nn.softmax(logits, name = 'Y_proba')

# cost functions
learning_rate = 0.01
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
loss = tf.reduce_mean(xentropy, name = 'loss')

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name = 'training_op')

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = 'accuracy')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# fetch MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:\\Users\\maliqi\\Desktop\\tensorflow\\MNIST_data')

X_train1 = mnist.train.images[mnist.train.labels < 5]
y_train1 = mnist.train.labels[mnist.train.labels < 5]
X_valid1 = mnist.validation.images[mnist.validation.labels < 5]
y_valid1 = mnist.validation.labels[mnist.validation.labels < 5]
X_test1 = mnist.test.images[mnist.test.labels < 5]
y_test1 = mnist.test.labels[mnist.test.labels < 5]


# execution phase
n_epochs = 400
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
	init.run()

	for epoch in range(n_epochs):
		rnd_idx = np.random.permutation(len(X_train1))
		for rnd_indices in np.array_split(rnd_idx, len(X_train1) // batch_size):
			X_batch, y_batch = X_train1[rnd_indices], y_train1[rnd_indices]
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
		loss_val, acc_val = sess.run([loss, accuracy], feed_dict = {X:X_valid1, 
			y:y_valid1})
		if loss_val < best_loss:
			save_path = saver.save(sess, './my_mnist_model_0_to_4.ckpt')
			best_loss = loss_val
			checks_without_progress = 0
		else:
			checks_without_progress += 1
			if checks_without_progress > max_checks_without_progress:
				print('Early stopping!')
				break
		print('{}\tValidaion loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%'.format(
			epoch, loss_val, best_loss, acc_val * 100))
with tf.Session() as sess:
	saver.restore(sess, './my_mnist_model_0_to_4.ckpt')
	acc_test = accuracy.eval(feed_dict = {X:X_test1, y:y_test1})
	print('Final test accuracy: {:.2f}%'.format(acc_test * 100))


from dnn_classifier import DNNClassifier
dnn_clf = DNNClassifier(random_state = 42)
dnn_clf.fit(X_train1, y_train1, n_epochs=1000,
	X_valid=X_valid1, y_valid=y_valid1)

from sklearn_metrics import accuracy_score
print('acc: ', accuracy_score(y_test1, Y_pred))