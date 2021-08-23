import numpy as np

def OR(x1, x2):                    #OR 게이트
    x = np.array([x1, x2])
    w = np.array([0.5,0.5])        #AND와는 가중치(w, b)만 다름
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1