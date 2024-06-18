import matplotlib.pyplot as plt
import numpy as np
from utils import utils

x = np.linspace(-5, 5, 100)  # -5から5までの範囲を100分割したnumpy配列
y = utils.normal(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
