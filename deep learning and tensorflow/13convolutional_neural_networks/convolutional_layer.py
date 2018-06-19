import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

def plot_image(image):
	plt.imshow(image, cmap = 'gray', interpolation = 'nearest')
	plt.axis('off')

def plot_color_image(image):
	plt.imshow(image.astype(np.uint8), interpolation = 'nearest')
	plt.axis('off')

china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
image = china[150:220, 130:250]
height, width, channels = image.shape
image_grayscale = image.mean(axis = 2).astype(np.float32)
images = image_grayscale.reshape(1, height, width, 1)

fmap = np.zeros(shape = (7, 7, 1, 2), dtype = np.float32)
fmap[:, 3, 0, 0] = 1
fmap[3, :, 0, 1] = 1
fmap[:, :, 0, 0]
plot_image(fmap[:, :, 0, 0])
plt.show()
plot_image(fmap[:, :, 0, 1])
plt.show()