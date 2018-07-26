from keras.engine.training import Model
from keras.layers import Input, Dense, Dropout, GRU, Bidirectional, merge, Embedding, concatenate, BatchNormalization, \
    Lambda
from keras.layers.cudnn_recurrent import CuDNNGRU
from keras.regularizers import l2
import numpy as np

from layers.BidirectionalRNN import BidirectionalRNN
from layers.Bilinear import Bilinear
from layers.NormalAttention import NormalAttention
from layers.GatedAttention import GatedAttention
from layers.selfattention import SelfAttention


class IAM(Model):
    def __init__(self, unit=64, dropout=0.2, max_len=39, update_num=3, regularization=0.1,
                 embedding_matrix=None, use_cudnn=False, use_share=False, use_one_cell=False):
        self.unit = unit
        self.dropout = dropout
        self.use_share = use_share
        self.use_one_cell = use_one_cell
        self.regularization = l2(regularization)

        Q1_input = Input(shape=(max_len,), dtype='int32', name='Q1')  # (?, L)
        Q2_input = Input(shape=(max_len,), dtype='int32', name='Q2')  # (?, L)
        # Q1_m = Input(shape=(max_len,), dtype='int32', name='mask1')
        # Q2_m = Input(shape=(max_len,), dtype='int32', name='mask2')
        # magic = Input(shape=(4,), dtype='float32', name='magic')

        embedding = Embedding(input_dim=embedding_matrix.shape[0],
                              output_dim=embedding_matrix.shape[1],
                              mask_zero=True,
                              weights=[embedding_matrix],
                              trainable=False)
        # bn = BatchNormalization()
        Q1 = embedding(Q1_input)
        Q2 = embedding(Q2_input)

        GRULayer = CuDNNGRU if use_cudnn else GRU
        for i in range(update_num):
            Q1, Q2 = self.update_module(Q1, Q2, GRULayer)
        Q1, Q2 = self.attention(Q1, Q2, GRULayer, implementation=3)
        # bn1 = BatchNormalization()
        # bnm = BatchNormalization()
        # bns = BatchNormalization()
        # regression = Bilinear(implementation=0, activation='tanh')([Q1, Q2])
        att = SelfAttention(1, activation='tanh')
        Q1 = att(Q1)
        Q2 = att(Q2)
        vector = concatenate([  # Q1, Q2,
                              merge.multiply([Q1, Q2]),
                              # merge.subtract([Q1, Q2], use_abs=True),
                              merge.subtract([Q1, Q2]),
                              merge.average([Q1, Q2])
                              ])
        # vector = merge.subtract([Q1, Q2])
        # vector = merge.add([Q1, Q2])
        # vector = Dropout(self.dropout)(vector)
        # vector = Dense(units=512, activation='tanh')(vector)
        # magic_new = Dense(units=64, activation='tanh')(magic)

        # vector = concatenate([vector, magic_new])
        vector = Dropout(self.dropout)(vector)

        vector = Dense(units=256, activation='tanh')(vector)

        vector = Dropout(self.dropout)(vector)

        regression = Dense(units=1, activation='sigmoid')(vector)
        super(IAM, self).__init__(inputs=[Q1_input, Q2_input], outputs=regression)

    def update_module(self, Q1, Q2, GRU):
        if self.use_share:
            if self.use_one_cell:
                bigru = BidirectionalRNN(GRU(self.unit, return_sequences=True,
                                             return_state=False,
                                             # dropout=self.dropout, recurrent_dropout=self.dropout,
                                             kernel_regularizer=self.regularization,
                                             recurrent_regularizer=self.regularization
                                             ))
            else:
                bigru = Bidirectional(GRU(self.unit, return_sequences=True,
                                          return_state=False,
                                          # dropout=self.dropout, recurrent_dropout=self.dropout,
                                          kernel_regularizer=self.regularization,
                                          recurrent_regularizer=self.regularization
                                          ))
            Q1 = bigru(Q1)
            Q2 = bigru(Q2)
        else:
            bigru1 = Bidirectional(GRU(self.unit, return_sequences=True,
                                       return_state=False, dropout=self.dropout,
                                       recurrent_dropout=self.dropout,
                                       kernel_regularizer=self.regularization,
                                       recurrent_regularizer=self.regularization
                                       ))
            bigru2 = Bidirectional(GRU(self.unit, return_sequences=True,
                                       return_state=False, dropout=self.dropout,
                                       recurrent_dropout=self.dropout,
                                       kernel_regularizer=self.regularization,
                                       recurrent_regularizer=self.regularization
                                       ))
            Q1 = bigru1(Q1)
            Q2 = bigru2(Q2)
        Q1, Q2 = GatedAttention(units=1)([Q1, Q2])
        return Q1, Q2

    def attention(self, Q1, Q2, GRU, implementation=0):
        if self.use_share:
            if self.use_one_cell:
                bigru = BidirectionalRNN(GRU(self.unit, return_sequences=True,
                                             return_state=True,
                                             # dropout=self.dropout, recurrent_dropout=self.dropout,
                                             kernel_regularizer=self.regularization,
                                             recurrent_regularizer=self.regularization
                                             ))
            else:
                bigru = Bidirectional(GRU(self.unit, return_sequences=True,
                                          return_state=True,
                                          # dropout=self.dropout, recurrent_dropout=self.dropout,
                                          kernel_regularizer=self.regularization,
                                          recurrent_regularizer=self.regularization
                                          ))

            Q1, forward_state1, backward_state1 = bigru(Q1)
            Q2, forward_state2, backward_state2 = bigru(Q2)
        else:
            bigru1 = Bidirectional(GRU(self.unit, return_sequences=True,
                                       return_state=True, dropout=self.dropout,
                                       recurrent_dropout=self.dropout,
                                       kernel_regularizer=self.regularization,
                                       recurrent_regularizer=self.regularization
                                       ))
            bigru2 = Bidirectional(GRU(self.unit, return_sequences=True,
                                       return_state=True, dropout=self.dropout,
                                       recurrent_dropout=self.dropout,
                                       kernel_regularizer=self.regularization,
                                       recurrent_regularizer=self.regularization
                                       ))

            Q1, forward_state1, backward_state1 = bigru1(Q1)
            Q2, forward_state2, backward_state2 = bigru2(Q2)
        last_output1 = merge.concatenate([forward_state1, backward_state1])
        last_output2 = merge.concatenate([forward_state2, backward_state2])
        # print(last_output1, last_output2)
        if implementation == 0:
            Q1 = NormalAttention()([last_output2, Q1])
            Q2 = NormalAttention()([last_output1, Q2])
        elif implementation == 1:
            att = NormalAttention()
            Q1 = att([last_output2, Q1])
            Q2 = att([last_output1, Q2])
        elif implementation == 2:
            Q1 = last_output1
            Q2 = last_output2
        elif implementation == 3:
            return Q1, Q2
        else:
            raise ValueError('implementation value error')

        return Q1, Q2


if __name__ == '__main__':
    array = np.random.randn(300, 200)
    model = IAM(embedding_matrix=array, use_share=True, regularization=0, use_one_cell=True)



