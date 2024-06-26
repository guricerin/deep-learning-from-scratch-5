import matplotlib.pyplot as plt
import numpy as np

X = np.array([[-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2],
              [-2, -1, 0, 1, 2]])
Y = np.array([[-2, -2, -2, -2, -2],
              [-1, -1, -1, -1, -1],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [2, 2, 2, 2, 2]])
Z = X ** 2 + Y ** 2
ax = plt.axes(projection='3d') # 3D用グラフを描画
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

xs = np.arange(-2, 2, 0.1) # -2から2まで0.1刻み
ys = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(xs, ys) # 格子点を作成
Z = X ** 2 + Y ** 2
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

ax = plt.axes()
ax.contour(X, Y, Z) # 等高線を描画
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
