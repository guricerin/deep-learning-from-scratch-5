import os

import matplotlib.pyplot as plt
import numpy as np

path = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
xs = np.loadtxt(path)

print(xs.shape)
print(xs[0])

plt.scatter(xs[:, 0], xs[:, 1])
plt.xlabel("Eruption time (min)") # 噴出時間（分）
plt.ylabel("Waiting time (min)") # 次の噴出までの時間（分）
plt.show()
