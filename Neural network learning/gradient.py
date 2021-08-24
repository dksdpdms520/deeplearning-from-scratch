# 경사법: 신경망 학습에 많이 쓰임 (경사 하강법)

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

 # 인수 f: 최적화하려는 함수
 # init_x: 초깃값
 # lr: 학습률 / 학습률을 적절히 설정하는 것이 중요
 # 하이퍼파라미터: 학습률 같은 매개변수 / 사람이 직접 설정해야 하는 매개변수
 # numerical_gradient(f,x): 함수의 기울기
 # step_num: 기울기에 학습률을 곱한 값으로 갱신하는 처리 반복 횟수

# 신경망 학습 절차
# 1단계 미니배치: 훈련 데이터 중 일부를 무작위로 가져옴. 미니배치의 손실 함수 값을 줄이는 것이 목표
# 2단계 기울기 산출: 미니배치의 손실 함수 값을 줄이기 위해 각 가중치 매개변수의 기울기 구함.
# 3단계 매개변수 갱신: 가중치 매개변수를 기울기 방향으로 아주 조금 갱신
# 4단계 반복


import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)