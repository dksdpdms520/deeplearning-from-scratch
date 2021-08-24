import numpy as np

# NAND게이트

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5,-0.5])       # AND와는 가중치(w, b)만 다름
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1