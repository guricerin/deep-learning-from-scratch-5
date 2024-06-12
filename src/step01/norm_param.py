import matplotlib.pyplot as plt
import numpy as np
from step01 import common

x = np.linspace(-10, 10, 1000)

y0 = common.nomal(x, mu=-3)
y1 = common.nomal(x, mu=0)
y2 = common.nomal(x, mu=5)

plt.plot(x, y0, label="$\mu$=-3")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
