import matplotlib.pyplot as plt
import numpy as np

# === 学習済みのパラメータ ===
mus = np.array([[2.0, 54.50],
                [4.3, 80.0]])
covs = np.array([[[0.07, 0.44], [0.44, 33.7]],
                 [[0.17, 0.94], [0.94, 36.00]]])
phis = np.array([0.35, 0.65])
# ==========================

def sample():
    z = np.random.choice(2, p=phis) # phisで指定された確率分布に従って0か1を選択
    mu, cov = mus[z], covs[z]
    x = np.random.multivariate_normal(mu, cov) # 多次元正規分布からデータを1つサンプリング
    return x

N = 500
xs = np.zeros((N, 2))
for i in range(N):
    xs[i] = sample()

plt.scatter(xs[:, 0], xs[:, 1], color="orange", alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
