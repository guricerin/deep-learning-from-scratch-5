import matplotlib.pyplot as plt
import numpy as np

from step01 import common


def sample_sum(sample_size):
    x_sums = []
    N = sample_size
    for _ in range(10000):
        xs = []
        for _ in range(N):
            x = np.random.rand()
            xs.append(x)
        t = np.sum(xs)
        x_sums.append(t)
    x_norm = np.linspace(-5, 5, 1000)
    mu = N / 2
    sigma = np.sqrt(N / 12)
    y_norm = common.nomal(x_norm, mu=mu, sigma=sigma)

    plt.hist(x_sums, bins="auto", density=True)
    plt.plot(x_norm, y_norm)
    plt.title(f"N={N}")
    plt.xlim(-1, 6)  # x軸の範囲を指定
    plt.show()


sample_sum(5)
