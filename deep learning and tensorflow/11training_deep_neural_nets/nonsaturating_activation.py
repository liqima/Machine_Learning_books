import numpy as np
import matplotlib.pyplot as plt


def relu(z):
	return np.maximum(0, z)

def leaky_relu(z, alpha = 0.01):
	return np.maximum(alpha * z, z)

z = np.linspace(-5, 5, 200)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([0, 0], [5, -1], 'k-')
plt.plot([-5, 5], [5, 5], 'k-')
plt.plot([-5, 5], [-1, -1], 'k-')
# plt.plot(z, relu(z), 'b-', linewidth = 2)
plt.plot(z, leaky_relu(z, 0.05), 'b-', linewidth = 2)
plt.grid(True)
props = dict(facecolor = 'black', shrink = 0.1)
plt.annotate('Leaky', xytext = (-3.5, 0.5), xy = (-5, -0.2), 
	arrowprops = props, fontsize = 14, ha = 'center')
plt.title('ReLu activation function', fontsize = 14)
plt.axis([-5, 5, -0.5, 4.2])

plt.show()