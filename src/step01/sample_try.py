import numpy as np

N = 3  # サンブルサイズ

xs = []
for _ in range(N):
    x = np.random.rand()  # 0以上1未満の乱数を一様分布から生成
    xs.append(x)

x_mean = np.mean(xs)
print(x_mean)
