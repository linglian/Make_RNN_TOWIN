#coding=utf-8
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.optimizers import RMSprop
from keras import losses
from random import random
import pandas as pd
import numpy as np
import sys
import datetime
import tensorflow as tf

model = Sequential()
model.add(SimpleRNN(input_dim=1, units=300, return_sequences=True))
model.add(SimpleRNN(input_dim=300, units=500, return_sequences=True))
model.add(SimpleRNN(input_dim=500, units=1000, return_sequences=True))
model.add(SimpleRNN(input_dim=1000, units=300))
model.add(Dense(units=1, input_dim=300))
model.add(Activation('linear'))
rmsprop = RMSprop(lr=0.0001)
model.load_weights('./my_model_weight.h5')
model.compile(loss=losses.mean_squared_error, optimizer=rmsprop,
              metrics=['mae', 'acc'])

def _load_data(data, n_prev = 2000):
    docX, docY = [], []
    for i in range(len(data) - n_prev - 1):
        docX.append(data[i : i + n_prev])
        docY.append(data[i + n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def _load_data2(data, n_prev = 100):
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data.iloc[i : i + n_prev].as_matrix())
        docY.append(data.iloc[i + n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    
    return alsX, alsY

def train_test_split(df, test_size=0.):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    print ntrn
    x_train, y_train = _load_data(df[0:ntrn])
    print x_train.shape
    x_test, y_test = _load_data(df[ntrn:])
    print x_test.shape
    return (x_train, y_train), (x_test, y_test)

data = np.load('../龙虎斗平民场_2018-05-28_16:58:12.npy')

result_list = []

#for i in data2:
#    result_list.insert(0, [i])
    
for i in data:
    result_list.insert(0, i)

data = np.array(result_list)


(x_train, y_train), (x_test, y_test) = train_test_split(data)

#for idx, x in enumerate(x_train):
#    print x_train[idx][-1], y_train[idx]

model.fit(x_train, y_train, nb_epoch=1, validation_split=0.05, steps_per_epoch=1000)

model.save_weights('../my_model_weight_{}.h5'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))