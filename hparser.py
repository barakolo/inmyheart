from consts import *
import numpy as np
import pandas as pd


def ds2type(ds_name):
    for t in types:
        if t in ds_name:
            return types[t]
    return None


class Signal():
    def __init__(self, sig_arr, type):
        self._type = type
        self._sample_size = len(sig_arr)
        self._sig_arr = sig_arr
        self._vec_arr = np.array(sig_arr, dtype=np.float64)

    def get_type(self):
        return self._type

    def get_signal_size(self):
        return self._signal_size

    def get_sig_arr(self):
        return self._sig_arr

    def as_vec(self):
        return self._vec_arr


def get_sigs(df, type):
    '''
    :param df: gets a long 1d signal array contains multiple samples sequentially.
    :return: extracts samples, each of size SIGNAL_SIZE, puts them in ordered matrix.
    '''
    xs = pd.DataFrame()
    ttl_len = len(df) - (len(df) % SIGNAL_SIZE)
    samples_num = ttl_len // SIGNAL_SIZE
    ys = pd.DataFrame([type]*samples_num, columns=['y'])
    for i in range(0, ttl_len, SIGNAL_SIZE):
        cur_sig = df[i:i+SIGNAL_SIZE].transpose()
        cur_sig = cur_sig.rename(lambda x: 'x' + str(x % SIGNAL_SIZE), axis='columns')
        #print('sig%d' % i, cur_sig.shape)
        xs = xs.append(cur_sig)
    return xs, ys

def sigs2vecs(sigs_arr):
    samples_cnt = len(sigs_arr)
    m = np.mat((SIGNAL_SIZE + 1, samples_cnt), dtype=np.float64)
    for i in range(len(sigs_arr)):
        cur_sig = sigs_arr[i]
        m[:-1, i] = cur_sig.as_vec()
        m[-1, i] = cur_sig.get_type()
    return m

