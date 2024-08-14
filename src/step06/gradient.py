# gradient: 勾配
# ある点における関数の微分をベクトルにしたもの
# その関数の最大値（かもしれない）方向を指し示すことになる
# 勾配にマイナスをかければ、最小値（かもしれない）方向を指し示すことになる
# 勾配法: 勾配の方向にある距離だけ進み、その地点で再度勾配を求めることを繰り返すことで、関数の最大値 or 最小値を求める方法

import torch


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

lr = 0.001 # 学習率
iters = 10000 # 繰り返し回数

for i in range(iters):
    if i % 1000 == 0:
        print(x0.item(), x1.item())

    y = rosenbrock(x0, x1)
    y.backward()

    # 値の更新
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

    # 勾配のリセット
    x0.grad.zero_()
    x1.grad.zero_()

print(x0.item(), x1.item())
