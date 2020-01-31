import numpy as np
import time, math
import tensorflow as tf


def data_preProcess(x_train,y_train,x_test,y_test):
  len_train, len_test = len(x_train), len(x_test)
  y_train = y_train.astype('int64').reshape(len_train)
  y_test = y_test.astype('int64').reshape(len_test)

  train_mean = np.mean(x_train, axis=(0,1,2))
  train_std = np.std(x_train, axis=(0,1,2))

  normalize = lambda x: ((x - train_mean) / train_std).astype('float32') # todo: check here
  pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

  x_train = normalize(pad4(x_train))
  x_test = normalize(x_test)

  return x_train,y_train,x_test,y_test
