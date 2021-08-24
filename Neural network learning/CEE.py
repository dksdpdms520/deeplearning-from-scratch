 # 손실 함수: 신경망 학습에서 사용하는 지표
 # 오차제곱합: 가장 많이 쓰이는 손실 함수

def cross_entropy_error(y,t):
    delta = 1e-7 # 마이너스 무한대가 발생하지 않도록 delta 값 더함
    return -np.sum(t * np.log(y + delta))