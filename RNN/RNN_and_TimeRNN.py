# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import sigmoid

# RNN: 순환신경망, 은닉상태 기억함
class RNN: # RNN 구현 
    def __init__(self, Wx, Wh, b): # (가중치, 가중치, 편향)
        self.params = [Wx, Wh, b] 
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] # 각 매개변수에 대응하는 형태로 기울기를 초기화한 후 저장
        self.cache = None # 역전파 계산 시 사용하는 중간 데이터를 담을 cache를 None으로 초기화

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN: # TimeRNN 구현 
    def __init__(self, Wx, Wh, b, stateful=False): #(가중치, 가중치, 편향,
        # stateful=False: TimeRNN 계층의 은닉 상태를 0행렬: 무상태 모드 / stateful=True: TimeRNN 계층의 은닉 상태 유지)
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params # 가중치, 가중치, 편향
        N, T, D = xs.shape # 미니배치 크기, 횟수, 입력 벡터의 차원 수 = xs의 형상
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None: # 원소가 모두 0인 행렬로 초기화
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs): # dhs: 상류(출력층)에서부터 전해지는 기울기
        Wx, Wh, b = self.params # 가중치, 가중치, 편향
        N, T, H = dhs.shape # 미니배치 크기, 횟수, 입력 벡터의 차원 수 = dhs의 형상
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f') # dxs :하류로 내보내는 기울기
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

# RNN: 순환하는 경로, 은닉 상태 기억
# BPTT: RNN의 순환 경로를 펼침으로써 다수의 RNN 계층이 연결된 신경망으로 해석, 보통의 오차역전파법으로 학습
# Truncated BPTT: 긴 시계열 데이터를 학습할 때 데이터를 적당한 길이씩 모으고, 블록 단위로 BPTT 학습을 수행 / 역전파 연결만 끊고 순전파 연결 유지: 데이터를 순차적으로 입력
# 언어 모델은 단어 시퀀스를 확률로 해석