import matplotlib.pyplot as plt
import numpy as np
from step02 import common

xs = common.load_sample_height_data()

mu = np.mean(xs)
sigma = np.std(xs)
# 新たなサンプルを生成
samples = np.random.normal(mu, sigma, 10000)

plt.hist(xs, bins="auto", density=True, alpha=0.7, label="original")
plt.hist(samples, bins="auto", density=True, alpha=0.7, label="generated")
plt.xlabel("Height(cm)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
