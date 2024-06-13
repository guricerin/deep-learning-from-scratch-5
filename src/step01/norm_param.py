import matplotlib.pyplot as plt
import numpy as np

from step01 import common

x = np.linspace(-10, 10, 1000)

### 平均を変えた場合、確率密度の最大値となる位置が変わる
y0 = common.nomal(x, mu=-3)
y1 = common.nomal(x, mu=0)
y2 = common.nomal(x, mu=5)
plt.plot(x, y0, label="$\mu$=-3")
plt.plot(x, y1, label="$\mu$=0")
plt.plot(x, y2, label="$\mu$=5")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

### 標準偏差が大きい（分散が大きい）ほど、確率密度の最大値が小さくなり、グラフが末広がりになる
y0 = common.nomal(x, mu=0, sigma=0.5)
y1 = common.nomal(x, mu=0, sigma=1)
y2 = common.nomal(x, mu=0, sigma=2)
plt.plot(x, y0, label="$\sigma$=0.5")
plt.plot(x, y1, label="$\sigma$=1")
plt.plot(x, y2, label="$\sigma$=2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
