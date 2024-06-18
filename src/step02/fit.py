import matplotlib.pyplot as plt
import numpy as np
from step02 import common
from utils import utils

xs = common.load_sample_height_data()
# 最尤推定: 正規分布のパラメータはサンプルの平均と標準偏差であると推定可能
mu = np.mean(xs)
sigma = np.std(xs)

print(f"mu: {mu}, sigma: {sigma}")

x = np.linspace(150, 190, 1000)
y = utils.normal(x, mu, sigma)

plt.hist(xs, bins="auto", density=True)
plt.plot(x, y)
plt.xlabel("Height(cm)")
plt.ylabel("Probability Density")
plt.show()
