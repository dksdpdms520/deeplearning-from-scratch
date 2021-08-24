 # 손실 함수: 신경망 학습에서 사용하는 지표
 # 오차제곱합: 가장 많이 쓰이는 손실 함수

def sum_squares_error(y,t): # 인수 y, t: 넘파이 배열
    return 0.5 * np.sum((y -t)**2)