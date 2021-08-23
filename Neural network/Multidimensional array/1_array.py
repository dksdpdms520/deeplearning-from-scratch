import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
# [1, 2, 3, 4]

np.ndim(A)                      # 배열의 차원 수 확인
# 1

A.shape                         # 배열의 형상 확인
# (4,)

A.shape[0]
# 4