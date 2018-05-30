#coding=utf-8

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import to_categorical
from keras.layers.recurrent import SimpleRNN, LSTM, RNN, GRU
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras import losses
from random import random
import keras
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import datetime

def _load_data(data, n_prev = 10):
    docX, docY = [], []
    for i in range(len(data) - n_prev - 1):
        docX.append(data[i : i + n_prev])
        docY.append(data[i + n_prev][0])
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0., n_prev):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    print ntrn
    x_train, y_train = _load_data(df[0:ntrn], n_prev)
    print x_train.shape
    x_test, y_test = _load_data(df[ntrn:], n_prev)
    print x_test.shape
    return (x_train, y_train), (x_test, y_test)

train_list = [
    './Make_RNN_TO_WIN/龙虎斗平民场_2018-05-30_20:42:04.npy',
#    './龙虎斗平民场_2018-05-28_21:21:59.npy',
#    './龙虎斗平民场_2018-05-29_23:15:02.npy'
]

step_list = [
#    25,
#    25,
    50
]



for _ in range(1000):

    prevs = [10, 25, 50, 100, 150, 200, 250, 300, 400, 500]

    for tii_in in range(len(prevs)):

        for ti_train in range(5):
            for idx, t_l in enumerate(train_list):
                data = np.load(t_l)

                if len(data) >= 20:

                    # print 'Start %s' % os.path.join('./', fi)

                    result_list = []

                    for i in data:
                        result_list.insert(0, i)

                    data = np.array(result_list)

                    (x_train, y_train), (x_test, y_test) = train_test_split(data, prevs[tii_in])

                    print y_train.shape
                    y_train = to_categorical(y_train)

                    lr = 0.00001
                    model = Sequential()
                    model.add(SimpleRNN(input_shape=(None, 1), units=100, return_sequences=True))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.0001))
                    model.add(SimpleRNN(input_shape=(None, 100), units=300, return_sequences=True))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.0001))
                    model.add(SimpleRNN(input_shape=(None, 300), units=500, return_sequences=True))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.0001))
                    model.add(SimpleRNN(input_shape=(None, 500), units=1000, return_sequences=True))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.0001))
                    model.add(SimpleRNN(input_shape=(None, 500), units=3000, return_sequences=False))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.0001))
                    model.add(Dense(input_shape=(None, 3000), units=3, activity_regularizer=keras.regularizers.l1(0.001),
                                    kernel_regularizer=keras.regularizers.l2(0.001), activation='softmax'))
                    model.add(Activation('relu'))
                    rmsprop = RMSprop(lr=lr)
                    model.load_weights('./my_model_weight_softmax3_.h5')
                    model.compile(loss=losses.categorical_crossentropy, optimizer=rmsprop,
                                  metrics=[keras.metrics.binary_accuracy])

                    try:
                        model.fit(x_train, y_train, batch_size=50, validation_split=1, epochs=1000, shuffle=True)
                    except Exception, ex:
                        print ' '
                        print ex
                    pre = model.evaluate(x=x_train, y=y_train)
                    for idx, i in enumerate(['acc']):
                        print '%s = %s' % (i, pre[idx])
                    model.save_weights('./my_model_weight_softmax3_.h5')
                    model.save_weights('./my_model_weight_softmax3_{}.h5'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))

model.save_weights('./my_model_weight_softmax3_.h5')