import numpy as np

# 소프트맥스 함수 구현
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c)             오버플로 대책: 입력 신호 중 최댓값을 뻄
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#
#     return y

a = np.array([1010, 1000, 900])
c = np.max(a)
np.exp(a -c) / np.sum(np.exp(a - c))
# array([9.99954600e-01, 4.539786e-05, 2.06106005e=09])

# 소프트맥스 함수: 출력 총 합 1

