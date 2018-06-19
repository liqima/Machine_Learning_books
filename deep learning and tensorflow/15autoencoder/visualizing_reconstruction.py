from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:/Users/maliqi/Desktop/tensorflow/MNIST_data')

import tensorflow as tf
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

saver = tf.train.Saver()
n_test_digits = 2
X_test = mnist.test.images[:n_test_digits]

with tf.Session() as sess:
    saver.restore(sess, "./my_model_weights.ckpt") # not shown in the book
    outputs_val = outputs.eval(feed_dict={X: X_test})

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

for digit_index in range(n_test_digits):
    plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
    plot_image(X_test[digit_index])
    plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
    plot_image(outputs_val[digit_index])