import librosa
import os
import numpy as np
import pydub
#from aip import AipSpeech

def get_feature(filename):
    y, sr = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = mfccs.T
    return mfccs


def process_mfccs(mfccs, type = 0, feature_len = 43):
    n, m = mfccs.shape
    # 43行特征值代表一秒，以一秒为基础单位
    k = int(n/feature_len)
    data = []
    for i in range(k):
        # mfcc = [mfccs[(i*feature_len):((i+1)*feature_len)]]
        mfcc = mfccs[(i*feature_len):((i+1)*feature_len)]
        data.append(mfcc)

    if type == 0:
        label = np.zeros(k)
    elif type == 1:
        label = np.ones(k)

    return data, label