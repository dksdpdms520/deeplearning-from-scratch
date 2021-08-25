import numpy as np


class SGD:
    """확률적 경사 하강법（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():cd 
            params[key] -= self.lr * grads[key]

# SGD: 최적의 매개변수의 기울기(미분)
# 비등방성 함수에서는 비효율적이므로 단점 개선: 모멘텀, AdaGrad, Adam
# lr: 학습률
# update(params, grads): SGD반복

class Momentum:
    """모멘텀 SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# 모멘텀: 운동량 / SGD보다 최적화 갱신 경로 안정적

class Nesterov:
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    # NAG는 모멘텀에서 한 단계 발전한 방법이다. (http://newsight.tistory.com/224)

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:
    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# AdaGrad: 각각의 매개변수에 적응적으로 학습률을 조정하면서 학습 진행
# 과거의 기울기를 제곱하여 더해가기 때문에 학습 진행될수록 갱신 강도가 약해지며 어느 순간 갱신량이 0이되어 이를 개선: RMSprop

class RMSprop:
    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# RMSprop: 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영: 지수이동평균

class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
            # 1e-7을 더해 0이 되는 것을 막아줌

# 가중치 초깃값 설정: Xavier 초깃값, He 초깃값 효과적
# 배치 정규화: 빠른 학습, 초깃값에 영향을 덜 받음

# 바른 학습을 위해
# 오버피팅: 매개변수가 많고 표현력이 높은 모델 / 훈련 데이터가 적음
# - 가중치 감소 / 드롭아웃: 뉴런을 임의적으로 삭제하면서 학습하는 방법

# 하이퍼파리미터의 최적화: 무작위로 샘플링하여 탐색이 효과적
# 0. 하이퍼파리미터 값의 범위 설정
# 1. 하이퍼파리미터 무작위로 샘플링
# 2. 샘플링한 하이퍼파리미터 값을 사용하여 학습, 검증데이터로 정확도 확인(에폭 작게)
# 3. 1번, 2번 과정 특정횟수 반복, 정확도의 결과를 보고 하이퍼파라미터의 범위 좁히기