# ニューラルネットワーク
# 非線形なデータを扱うための手法

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        y = self.linear1(x)
        y = F.sigmoid(y)
        y = self.linear2(y)
        return y

torch.manual_seed(0)
x = torch.rand(100, 1)
noise = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + noise

learning_rate = 0.2
iters = 10000

model = Model()
# SDG: 確率的勾配降下法
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(iters):
    y_pred = model(x)
    loss = F.mse_loss(y, y_pred)

    loss.backward()
    optimizer.step() # パラメータの更新
    optimizer.zero_grad() # 勾配のリセット

    if i % 1000 == 0:
        print(loss.item())

print(loss.item())
