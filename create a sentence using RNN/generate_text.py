# coding: utf-8
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
model.load_params('../ch06/Rnnlm.pkl')

# start 문자와 skip 문자 설정
start_word = 'you'                               # 첫 단어
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']                 # 샘플링하지 않을 단어
skip_ids = [word_to_id[w] for w in skip_words]

# 문장 생성
word_ids = model.generate(start_id, skip_ids)       # generate(): 단어 id를 배열 형태로 변환
txt = ' '.join([id_to_word[i] for i in word_ids])   # 단어 ID 배열을 문장으로 변환
txt = txt.replace(' <eos>', '.\n')
print(txt)
