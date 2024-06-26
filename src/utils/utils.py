import numpy as np


def normal(x, mu=0, sigma=1):
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

def multivariate_normal(x, mu, cov):
    """
    多変量正規分布の確率密度関数

    Args:
        x: 確率変数の列ベクトル
        mu: 平均の列ベクトル
        cov: 共分散行列
             対角要素が分散を表し、非対角要素が共分散を表す
             共分散が正であれば2つの変数は正の相関、負であれば2つの変数は負の相関にある

    Returns:
        y: xにおける確率密度（1 * 1 の行列なのでスカラ値とみなせる）
    """
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(mu)
    z = 1 / (np.sqrt((2 * np.pi) ** D * det))
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y
