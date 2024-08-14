# 線形回帰

import torch

# トイデータセット
torch.manual_seed(0)
x = torch.rand(100,1)
noise = torch.rand(100,1)
y = 2 * x + 5 + noise

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    # @ は行列積の演算子
    y = x @ W + b
    return y

def mean_squared_error(x0, x1):
    """
    平均二乗誤差
    """

    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N

lerning_rate = 0.1
iters = 100

# 損失関数
# y と y_hat (= Wx + b) との差を最小化する
for i in range(iters):
    y_hat = predict(x)
    loss = mean_squared_error(y, y_hat)

    loss.backward()

    # 勾配法によってパラメータを更新
    W.data -= lerning_rate * W.grad.data
    b.data -= lerning_rate * b.grad.data

    W.grad.zero_()
    b.grad.zero_()

    if i % 10 == 0:
        print(loss.item())

print(loss.item())
print('====')
print('W =', W.item())
print('b =', b.item())
