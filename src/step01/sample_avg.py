import matplotlib.pyplot as plt
import numpy as np


def sample_avg(sample_size):
    x_means = []
    N = sample_size

    for _ in range(10000):
        xs = []
        for _ in range(N):
            x = np.random.rand()
            xs.append(x)
        mean = np.mean(xs)
        x_means.append(mean)

    plt.hist(x_means, bins="auto", density=True)
    plt.title(f"N={N}")
    plt.xlabel("x")
    plt.ylabel("Probability Density")  # y軸は確率密度
    plt.xlim(-0.05, 1.05)
    plt.ylim(0, 5)
    plt.show()


sample_avg(1)  # ほぼ一様分布
sample_avg(2)
sample_avg(4)  # 中心極限定理により正規分布に近づく
sample_avg(10)
