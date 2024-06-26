import numpy as np
from utils import utils

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
print('---')
print(A.T) # 転置

A = np.array([[3, 4], [5, 6]])
d = np.linalg.det(A) # 行列式
print(d)

B = np.linalg.inv(A) # 逆行列
print(B)
print('---')
print(np.dot(A, B)) # 行列と逆行列の積 = 単位行列

x = np.array([[0], [0]]) # or np.array([0, 0])
mu = np.array([[1], [2]]) # or np.array([1, 2])
cov = np.array([[1, 0],
                [0, 1]])
y = utils.multivariate_normal(x, mu, cov)
print(y)            
