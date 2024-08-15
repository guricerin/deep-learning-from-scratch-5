import torch
import torchvision
import torchvision.transforms as transforms

# Image型をTensor型に変換する処理
transform = transforms.ToTensor()

# MNISTデータセットを読み込む
dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# for文でミニバッチを取り出せる
for x, label in dataloader:
    print('x shape:', x.shape)
    print('label shape:', label.shape)
    break # 0番目のミニバッチの情報のみ表示するため、ここで抜ける
