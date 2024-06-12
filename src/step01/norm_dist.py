import numpy as np


def nomal(x, mu=0, sigma=1):
    """
    正規分布の確率密度関数
    Args:
        x: 確率変数
        mu: 平均
        sigma: 標準偏差
    """
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return y
