# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# RNN의 문제점: 똑같은 가중치 행렬의 곱을 반복하기 때문
# Matmul 노드를 지날 때마다 바뀌는 역전파 기울기의 크기 변화

N = 2   # 미니배치 크기
H = 3   # 은닉 상태 벡터의 차원 수
T = 20  # 시계열 데이터의 길이

dh = np.ones((N, H)) # 초기화 (np.ones(): 모든 원소가 1인 행렬을 반환)

np.random.seed(3) # 재현할 수 있도록 난수의 시드 고정

Wh = np.random.randn(H, H) # 오버플로 발생할 가능성으로 Wh 초기 값을 변경
# Wh = np.random.randn(H, H) * 0.5  # 기울기 dh는 시간 크기에 비례하여 지수적으로 감소 = 기울기 소실 = 장기 의존 학습 불가능

norm_list = [] # 각 단계에서의 dh크기 리스트에 추가
for t in range(T): # 역전파 MatMul 노드 수 T 만큼 dh 갱신
    dh = np.dot(dh, Wh.T) 
    norm = np.sqrt(np.sum(dh**2)) / N # 평균 L2norm (각각 원소를 제곱해 모두 더하고 제곱근을 취한 값) 
    norm_list.append(norm)

print(norm_list)

# 그래프 그리기
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()
# 기울기 dh는 시간 크기에 비례하여 지수적으로 증가 = 기울기 폭발 = 오버플로 발생 (-> line 17로 변경)
