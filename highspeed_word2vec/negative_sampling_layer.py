# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W) # EmbeddingDot 계층 저장
        self.params = self.embed.params # 매개변수 저장
        self.grads = self.embed.grads # 기울기 저장
        self.cache = None # 순전파 시 계싼 결과를 잠시 유지

    def forward(self, h, idx): # 순전파
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1) # 내적 계산

        self.cache = (h, target_W)
        return out
# EmbeddingDot계층: 입력층에서 각각 대응하는 단어 id의 분산 표현을 추출하기 위해
# h: 은닉층 뉴런으로 EmbeddingDot 계층을 거쳐 sigmoid with loss 계층을 통과 / idx: 단어 id의 넘파이 배열 (미니배치 처리르 가정했기 떄문에 배열로 받음)

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size): # (단어목록, 확률분포에 '제곱'할 값, 부정적 예 샘플링을 수행하는 횟수)
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target): # target 인수로 지정한 단어를 긍정적 예로 해석하고 그 외 단어 id를 샘플링 (부정적인 예를 골라줌)
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
                # replace=false: 샘플링 시 중복을 미포함

        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)
            # replace = true: 샘플링 시 중복값 포함

        return negative_sample
# nagative sampling: 부정적인 예 샘플링 (말뭉치의 단어 빈도, 확률 분포를 기준으로 샘플링)
# 부정적인 예:0에 가깝게 긍정적인 예: 1에 가깝게


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5): # (출력 측 가중치, 말뭉치 리스트, 확률분포에 제곱할 값, 부정적 예의 샘플링 횟수)
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        # sample_size + 1: 브정적 예를 다루는 계층인 sample_size에 긍정적 예를 다루는 계층 1개가 더 필요
        # loss_layers[0], embed_dot_layers[0]: 긍정적 예를 다루는 계층

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        # 매개변수와 기울기를 각각 배열로 저장

    def forward(self, h, target): # (은닉층 뉴런 h, 긍정적 예의 타깃)
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target) # 부정적 예를 샘플링하고 저장

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target) # EmbeddingDot계층의 순전파 점수
        correct_label = np.ones(batch_size, dtype=np.int32) # 이 점수를 sigmoid with loss 계층으로 흘러 손실 구함
        loss = self.loss_layers[0].forward(score, correct_label) # 긍정적 예: 1

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label) # 부정적 예: 0

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore) # 여러 개의 기울기 값을 더함

        return dh
