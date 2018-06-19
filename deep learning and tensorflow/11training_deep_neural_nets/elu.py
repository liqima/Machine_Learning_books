import matplotlib.pyplot as plt
import numpy as np

def elu(z, alpha = 1):
	return np.where(z < 0, alpha * (np.exp(z) - 1), z)

z = np.linspace(-5, 5, 200)
plt.plot(z, elu(z), 'b-', linewidth = 2)
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [-1, -1], 'k--')
plt.grid(True)

plt.show()	


