import numpy as np

A = np.array([[1, 2, 3], [3, 4, 6]])
A.shape
# (2, 3)

B = np.array([[1, 2], [3, 4], [5, 6]])
B.shape
# (3, 2)

np.dot(A,B)                           # 두 행렬의 곱 넘파이 함수
# array([[22,28],
#        [49, 64]])

# 행렬의 형상이 다를 때: 행렬 A의 1번째 차원의 열 수 = 행렬 B의 0번째 차원의 행 수