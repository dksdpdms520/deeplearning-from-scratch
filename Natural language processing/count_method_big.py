# coding: utf-8
# PTB 데이터셋 평가

import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb


window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('동시발생 수 계산 ...')
# 동시발생 행렬을 PPMI 행렬로 변환하고 다시 차원을 감소시킴으로써 거대한 희소벡터를 밀집벡터로 변환
C = create_co_matrix(corpus, vocab_size, window_size)
print('PPMI 계산 ...')
W = ppmi(C, verbose=True) 

print('calculating SVD ...')
try:
    # truncated SVD (빠르다!)
    from sklearn.utils.extmath import randomized_svd # 무작위 수를 사용한, 특잇값이 큰 것들만 계산하여 기본적인 SVD보다 빠름
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (느리다)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota'] # 검색어의 단어의 의미, 문법적 관점에서 비슷한 단어들을 가까운 벡터로 나타냄
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
