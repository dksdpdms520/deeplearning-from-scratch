# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel

# seq2seq: 한 시계열 데이터를 다른 시계열 데이터로 변환, 2개의 RNN을 사용, Encoder-Decoder
# Encoder: LSTM 계층, Embedding 계층으로 이루어짐 / LSTM의 은닉 상태만 Decoder에 전달
# Decoder: Time Embedding, Time LSTM, Time Affine 계층으로 이루어짐

class Encoder:  # Encoder
    def __init__(self, vocab_size, wordvec_size, hidden_size):  
        # 초기화 (문자의 종류, 문자 벡터의 차원 수, LSTM 계층 은닉 상태 벡터의 차원 수)
        # 가중치 매개변수 초기화, 필요한 계층 생성
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params  # 가중치 매개변수 저장
        self.grads = self.embed.grads + self.lstm.grads     # 기울기 저장
        self.hs = None

    def forward(self, xs):  # 순전파
        xs = self.embed.forward(xs)  
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh):  # 역전파 / dh: Decoder가 전해주는 기울기
        dhs = np.zeros_like(self.hs)  # dhs: 원소가 모두 0
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class Decoder:  # Decoder: Encoder 출력한 h 받아 목적으로 하는 다른 문자열 출력
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    # softmax with loss 계층에서 기울기를 받음
    def backward(self, dscore):  
        dout = self.affine.backward(dscore)   # Time affine 계층
        dout = self.lstm.backward(dout)       # Time LSTM 계층
        dout = self.embed.backward(dout)      # Time embedding 계층 순으로 기울기 전파
        dh = self.lstm.dh                     # dh: Time LSTM 계층의 시간 방향으로의 기울기 저장
        return dh

    def generate(self, h, start_id, sample_size):     # (Encoder로 부터 받은 은닉 상태, 최초 문자 id, 생성하는 문자 수)
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)          # affine 계층이 출력하는 점수가 가장 큰 문자 id를 선택하는 작업 반복

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseModel):  # Encoder와 Decoder를 연결, time softmax with loss 계층을 이용해 손실 계산
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
