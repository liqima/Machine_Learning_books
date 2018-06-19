import numpy as np
import matplotlib.pyplot as plt

def logit(z):
	return 1 / (1 + np.exp(-z))

z = np.linspace(-5, 5, 200)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, logit(z), 'b-', linewidth = 2)
props = dict(facecolor = 'black', shrink = 0.1)
plt.annotate('Saturating', xytext = (3.5, 0.7), xy = (5, 1), 
	arrowprops = props, fontsize = 14, ha = 'center')
plt.annotate('Saturating', xytext = (-3.5, 0.3), xy = (-5, 0),
	arrowprops = props, fontsize = 14, ha = 'center')
plt.annotate('Linear', xytext = (2, 0.2), xy = (0, 0.5), 
	arrowprops = props, fontsize = 14, ha = 'center')
plt.grid(True)
plt.title('Sigmoid activation function', fontsize = 14)
plt.axis([-5, 5, -0.2, 1.2])

plt.show()
