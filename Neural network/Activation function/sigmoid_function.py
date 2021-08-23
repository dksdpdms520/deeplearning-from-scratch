import numpy as np
import matplotlib.pylab as plt

# 시그모이드 함수 구현
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)                      # y축 범위 지정
plt.show()

# 시그모이드 = S자 모양
# 신경망: 활성화 함수 비선형 함수의 사용

# 퍼셉트론: 뉴런(노드)사이 0 또는 1이 흐름
# 신경망 (입력층, 은닉층, 출력층): 뉴런(노드)사이 연속적인 실수의 흐름 (입력이 아무리 작거나 커도 출력은 0과 1사이)