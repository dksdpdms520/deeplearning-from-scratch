import numpy as np

# 3층 신경망 구현

# 가중치와 편향을 초기화하고 딕셔너리 변수인 network에 저장
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.5])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([[0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([[0.1, 0.2])

    return network

# 입력 신호를 출력으로 변환하는 처리 과정
def forward(network, x):                                              # 순전파
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)                                         # 항등 함수 (identity_function): 입력을 그대로 출력하는 함수

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
# [0.31682708 0.69627909]

# 항등 함수: 회귀
# 소프트맥스 함수: 분류