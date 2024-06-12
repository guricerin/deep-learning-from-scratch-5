import matplotlib.pyplot as plt
import numpy as np


def nomal(x, mu=0, sigma=1):
    """
    正規分布の確率密度関数
    平均が0, 標準偏差が1の正規分布は標準正規分布と呼ばれる

    Args:
        x: 確率変数
        mu: 平均
        sigma: 標準偏差

    Returns:
        y: 確率密度
    """
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return y


x = np.linspace(-5, 5, 100)  # -5から5までの範囲を100分割したnumpy配列
y = nomal(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
