import matplotlib.pyplot as plt
import numpy as np

from step01 import common

x = np.linspace(-5, 5, 100)  # -5から5までの範囲を100分割したnumpy配列
y = common.nomal(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
