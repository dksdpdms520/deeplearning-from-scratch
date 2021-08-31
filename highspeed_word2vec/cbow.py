# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

# 단순한 SimpleCBOW 클래스 개선: embedding 계층, negative sampling loss 계층을 적용
# SimpleCBOW 클래스: 출력 측 가중치는 입력 측 가중치와 다른 형상으로 단어 벡터가 열 방향에 배치
# CBOW 클래스: 출력 측 가중치는 입력 측 가중치와 같은 형상으로 단어 벡터가 행 방향에 배치 (NegativeSamplingLoss 클래스에서 Embedding 계층을 사용하기 떄문)

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus): # (어휘 수, 은닉층 뉴런 수, 맥락의 크기, 단어 id 목록)
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)  # Embedding 계층 사용
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target): # (맥락, 타깃): 형태는 단어 id (SimpleCBOW와 다른 점으로 SimpleCBOW은 단어 id를 원핫 벡터로 변환하여 사용)
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
