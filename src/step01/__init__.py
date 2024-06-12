import matplotlib.pyplot as plt
import numpy as np

from step01 import common


def norm_dist():
    x = np.linspace(-5, 5, 100)  # -5から5までの範囲を100分割したnumpy配列
    y = common.nomal(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def norm_param():
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


def sample_try():
    N = 3  # サンブルサイズ

    xs = []
    for _ in range(N):
        x = np.random.rand()  # 0以上1未満の乱数を一様分布から生成
        xs.append(x)

    x_mean = np.mean(xs)
    print(x_mean)


def sample_avg_core(sample_size):
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


def sample_avg():
    sample_avg_core(1)  # ほぼ一様分布
    sample_avg_core(2)
    sample_avg_core(4)  # 中心極限定理により正規分布に近づく
    sample_avg_core(10)


def sample_sum_core(sample_size):
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


def sample_sum():
    sample_sum_core(5)
