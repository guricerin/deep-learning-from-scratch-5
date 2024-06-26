import numpy as np

# 3 * 1 のベクトル
x = np.array([1, 2, 3])

print(x.__class__)
print(x.shape) # 形状
print(x.ndim) # 次元数

# 2 * 3 の行列
W = np.array([[1, 2, 3],
              [4, 5, 6]])
print(W.shape)
print(W.ndim)

X = np.array([[0, 1, 2],
              [3, 4, 5]])
print('---')
print('W + X')
print(W + X)
print('---')
print('W * X')
print(W * X)
print('---')

# ベクトルの内積
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y = np.dot(a, b)
print(y)

# 行列の積
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
Y = np.dot(A, B)
print(Y)

