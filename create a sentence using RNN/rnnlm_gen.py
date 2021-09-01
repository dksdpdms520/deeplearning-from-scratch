# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

# 문장 생성 구현 (확률분포로부터 샘플링)

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):  # (최초로 주는 단어 id, 단어 id 리스트, 샘플링하는 단어 수)
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1) 
            score = self.predict(x)         # 각 단어의 정규화 되기 전의 값인 점수를 출력
            p = softmax(score.flatten())    # 점수들을 소프트맥스 함수를 이용하여 정규화

            # 확률분포 p값
            sampled = np.random.choice(len(p), size=1, p=p)       # 확률분포 p로부터 샘플링
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)