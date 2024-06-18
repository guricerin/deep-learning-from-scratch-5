import numpy as np
from scipy.stats import norm
from step02 import common

xs = common.load_sample_height_data()
mu = np.mean(xs)
sigma = np.std(xs)

# 160以下の値が発生する確率
p1 = norm.cdf(160, mu, sigma)
print("p(x <= 160):", p1)

p2 = norm.cdf(180, mu, sigma)
print("p(x > 180):", 1 - p2)
