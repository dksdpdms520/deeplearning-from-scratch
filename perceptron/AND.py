import numpy as np

def AND(x1, x2):                 #AND 게이트
    x = np.array([x1, x2])       #입력
    w = np.array([0.5, 0.5])     #가중치
    b = -0.7                     #편향
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

#y = 0 (b + w1x1 +w2x2 <= 0)
#y = 1 (b + w1x1 +w2x2 > 0)
#w1, w2: 가중치, b: 편향