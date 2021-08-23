import numpy as np
import matplotlib.pylab as plt

# ReLU 함수 구현

def relu(x):
    return np.maximum(0, x)

# ReLU 함수: 최근 주로 이용하는 활성화 함수

# 활성화 함수: 계단함수, 시그모이드 함수, ReLU 함수