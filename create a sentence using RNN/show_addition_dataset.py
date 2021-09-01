# coding: utf-8
import sys
sys.path.append('..')
from dataset import sequence


(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('addition.txt', seed=1984)   
    # load_data(file_name, seed): file_name으로 지정한 텍스트 파일을 읽어 텍스트 문자 id로 변환하고 훈련 데이터, 테스트 데이터로 나눠 반환
    # seed: 무작위수의 초깃값 (훈련 데이터와 테스트 데이터로 나누기 전 전체 데이터를 뒤섞을 때 무작위수 사용)
char_to_id, id_to_char = sequence.get_vocab()  # 문자와 문자 id의 대응 관계 상호 변환 가능
    # get_vocab(): 문자와 문자 id의 대응 관계를 담은 딕셔너리를 반환

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)
# (45000, 7) (45000, 5)
# (5000, 7) (5000, 5)

print(x_train[0])
print(t_train[0])
# [ 3  0  2  0  0 11  5]
# [ 6  0 11  7  5]

print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))
# 71+118
# _189
