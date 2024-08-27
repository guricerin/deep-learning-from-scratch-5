import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h) # 0.0 <= x_hat <= 1.0
        return x_hat

def reparameterize(mu, sigma):
    """
    変数変換トリック
    """
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z

class VAE(nn.Module):
    """
    Variational AutoEncoder（変分オートエンコーダー）
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction='sum')
        L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        return (L1 + L2) / batch_size

# ハイパーパラメータの設定
input_dim = 784 # 画像データxのサイズ（MNIST画像は28*28=784）
hidden_dim = 200 # ニューラルネットワークの中間層の次元数
latent_dim = 2 # 潜在変数ベクトルzの次元数
epochs = 30
learning_rate = 3e-4
batch_size = 32

# 画像データの前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(torch.flatten) # 画像データを1次元に変換
])
dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# モデルとオプティマイザ
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

# 学習ループ
for epoch in range(epochs):
    loss_sum = 0.0
    count = 0
    for x, label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        count += 1
    
    loss_avg = loss_sum / count
    losses.append(loss_avg)
    print(loss_avg)

# 新しい画像の生成
with torch.no_grad(): # with文内の勾配計算を計算を無効化し、メモリ使用量を削減する
    sample_size = 64
    z = torch.randn(sample_size, latent_dim) # 潜在空間の次元数（latent_dim）に従ってsample_size個のランダムな潜在変数zを生成
    x = model.decoder(z)
    generated_images = x.view(sample_size, 1, 28, 28)

# 生成した画像をグリッド状に並べる
grid_img = torchvision.utils.make_grid(
    generated_images,
    nrow=8,
    padding=2,
    normalize=True
)

plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()
