# coding: utf-8
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

# LSTM 계층을 사용한 RNNLM train

# LSTM 계층: RNN의 문제점 기울기 소실 개선
# 1. 기억 셀 (LSTM 계층에서만 데이터를 주고 받음)
# 2. 게이트 사용: 게이트에는 전용 가중치 존재, 시그모이드 함수를 사용하여 0.0 ~ 1.0 사이의 실수로 열림 상태 표현
# 3. output 게이트: 은닉상태 출력을 담당하는 게이트 / 아다마르 곱
# 4. forget 게이트: 이전 시각의 기억 셀로부터 잊어야 할 기억 삭제하는 게이트
# 5. 새로 기억해야 할 정보 기억 셀 g
# 6. input 게이트: g에 추가되는 게이트 (g의 각 원소가 새로 추가되는 정보로써 가치가 얼마나 큰지를 판단)
# 기억 셀에서 기울기 소실이 일어나지 않는 이유: forget 게이트에서 잊어야 한다고 판단한 기억 셀은 기울기를 작게 전하기 떄문

# 하이퍼파라미터 설정
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNN의 은닉 상태 벡터의 원소 수
time_size = 35     # RNN을 펼치는 크기
lr = 20.0          # 학습률
max_epoch = 4
max_grad = 0.25

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 모델 생성
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 기울기 클리핑을 적용하여 학습
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
            eval_interval=20) # 20번째 반복마다 퍼플렉서티를 평가 (데이터가 크기 때문에 모든 에폭에서 평가 아님)
# fit(): 모델의 기울기를 구해 모델의 매개변수를 갱신 (인수 max_grad: 기울기 클래핑 적용)
trainer.plot(ylim=(0, 500)) # 그래프

# 테스트 데이터로 평가
model.reset_state() # 모델 상태를 재설정하여 평가 (LSTM의 은닉상태와 기억 셀)
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 퍼플렉서티: ', ppl_test)

# 매개변수 저장
model.save_params()