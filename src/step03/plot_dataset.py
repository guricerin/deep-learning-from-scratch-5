import os

import matplotlib.pyplot as plt
import numpy as np

path = os.path.join(os.path.dirname(__file__), "height_weight.txt")
xs = np.loadtxt(path)
print(xs.shape)
small_xs = xs[:500]
plt.scatter(small_xs[:, 0], small_xs[:, 1]) # 散布図を描画
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()
