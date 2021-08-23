import numpy as np
import matplotlib.pylab as plt

# 단순한 계단 함수 구현
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 넘파이 배열 지원을 위한 계단 함수 구현
# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)

def step_function(x):
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)                              # y축 범위 지정
plt.show()